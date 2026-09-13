"""
AcademicAI Local Cost Calculation — Dynamic Model Pricing Cache & High-Precision Decimal Math.

Provides:
- ModelPricing: normalized per-token pricing structure derived from AcademicAI /api/v1/llm/models
- ModelPricingCache: thread-safe, TTL-based caching with graceful error fallback
- RequestCost: high-precision Decimal cost calculation per LLM request with standardized response headers
- calculate_request_cost: pure calculation function combining token usage and model pricing
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from starlette.concurrency import run_in_threadpool

from academicai.auth import get_base_url, get_headers
import academicai.config as config

log = logging.getLogger("academicai-proxy")


def _get_setting(name: str, explicit_value: Optional[Any] = None) -> Any:
    if explicit_value is not None:
        return explicit_value

    server_mod = sys.modules.get("server")
    if server_mod is not None and hasattr(server_mod, name):
        return getattr(server_mod, name)

    if hasattr(config, name):
        return getattr(config, name)

    return None


def _format_decimal(d: Decimal) -> str:
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s else "0"


@dataclass
class ModelPricing:
    def __init__(
        self,
        model_id: str,
        input_cost_per_token: Decimal,
        output_cost_per_token: Decimal,
        per_request_cost: Decimal = Decimal("0"),
        currency: Optional[str] = None,
        is_tiered: bool = False,
        raw_costs: Optional[list[dict[str, Any]]] = None,
    ):
        self.model_id = model_id
        self.input_cost_per_token = input_cost_per_token
        self.output_cost_per_token = output_cost_per_token
        self.per_request_cost = per_request_cost
        self._currency = currency
        self.is_tiered = is_tiered
        self.raw_costs = list(raw_costs or [])

    @property
    def currency(self) -> str:
        if self._currency is not None:
            return self._currency
        return get_pricing_cache().currency

    @currency.setter
    def currency(self, val: Optional[str]) -> None:
        self._currency = val

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model_id": self.model_id,
            "input_cost_per_token": str(self.input_cost_per_token),
            "output_cost_per_token": str(self.output_cost_per_token),
            "per_request_cost": str(self.per_request_cost),
            "is_tiered": self.is_tiered,
            "raw_costs": self.raw_costs,
        }
        if self._currency:
            d["currency"] = self._currency
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_currency: Optional[str] = None) -> "ModelPricing":
        curr = data.get("currency") or default_currency
        return cls(
            model_id=str(data.get("model_id", "")),
            input_cost_per_token=Decimal(str(data.get("input_cost_per_token", "0"))),
            output_cost_per_token=Decimal(str(data.get("output_cost_per_token", "0"))),
            per_request_cost=Decimal(str(data.get("per_request_cost", "0"))),
            currency=curr,
            is_tiered=bool(data.get("is_tiered", False)),
            raw_costs=list(data.get("raw_costs", [])),
        )

    def __repr__(self) -> str:
        return (
            f"ModelPricing(model_id={self.model_id!r}, "
            f"input_cost_per_token={self.input_cost_per_token}, "
            f"output_cost_per_token={self.output_cost_per_token}, "
            f"per_request_cost={self.per_request_cost}, "
            f"currency={self.currency!r}, "
            f"is_tiered={self.is_tiered})"
        )


def parse_model_costs(model_id: str, raw_costs: list[dict[str, Any]]) -> ModelPricing:
    """
    Parses AcademicAI's costs array.
    AcademicAI reports input_tokens and output_tokens costs per 1,000 tokens (1k tokens).
    Normalized rate per single token = Decimal(cost) / Decimal(1000).
    """
    input_rates: list[Decimal] = []
    output_rates: list[Decimal] = []
    per_request_rate = Decimal("0")

    for entry in raw_costs:
        if not isinstance(entry, dict):
            continue
        c_type = entry.get("costType")
        c_val = entry.get("cost")
        if c_val is None:
            continue
        try:
            d_val = Decimal(str(c_val))
        except Exception:
            continue

        if c_type == "input_tokens":
            input_rates.append(d_val / Decimal("1000"))
        elif c_type == "output_tokens":
            output_rates.append(d_val / Decimal("1000"))
        elif c_type == "per_request":
            per_request_rate = d_val

    is_tiered = len(input_rates) > 1 or len(output_rates) > 1
    input_per_token = input_rates[0] if input_rates else Decimal("0")
    output_per_token = output_rates[0] if output_rates else Decimal("0")

    return ModelPricing(
        model_id=model_id,
        input_cost_per_token=input_per_token,
        output_cost_per_token=output_per_token,
        per_request_cost=per_request_rate,
        currency=None,
        is_tiered=is_tiered,
        raw_costs=list(raw_costs),
    )


@dataclass
class RequestCost:
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    input_cost: Decimal
    output_cost: Decimal
    per_request_cost: Decimal
    request_cost: Decimal
    currency: str
    is_estimated: bool = False

    def to_headers(self) -> dict[str, str]:
        return {
            "X-AcademicAI-Request-Cost": _format_decimal(self.request_cost),
            "X-AcademicAI-Input-Cost": _format_decimal(self.input_cost),
            "X-AcademicAI-Output-Cost": _format_decimal(self.output_cost),
            "X-AcademicAI-Prompt-Tokens": str(self.prompt_tokens),
            "X-AcademicAI-Completion-Tokens": str(self.completion_tokens),
            "X-AcademicAI-Cost-Currency": self.currency,
            "X-AcademicAI-Cost-Estimated": "true" if self.is_estimated else "false",
        }


def calculate_request_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    pricing_map: Optional[dict[str, ModelPricing]] = None,
    pricing_cache: Optional["ModelPricingCache"] = None,
) -> RequestCost:
    """
    Calculates exact request-level cost using Decimal precision.
    """
    pricing: Optional[ModelPricing] = None
    cache = pricing_cache or get_pricing_cache()
    if pricing_map is not None:
        pricing = pricing_map.get(model)
    else:
        pricing = cache.get_pricing_with_refresh(model)

    p_tokens = max(0, int(prompt_tokens or 0))
    c_tokens = max(0, int(completion_tokens or 0))
    total = p_tokens + c_tokens
    currency = pricing.currency if pricing else cache.currency

    if pricing is None:
        return RequestCost(
            model=model,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            total_tokens=total,
            input_cost=Decimal("0"),
            output_cost=Decimal("0"),
            per_request_cost=Decimal("0"),
            request_cost=Decimal("0"),
            currency=currency,
            is_estimated=True,
        )

    p_dec = Decimal(p_tokens)
    c_dec = Decimal(c_tokens)

    input_cost = p_dec * pricing.input_cost_per_token
    output_cost = c_dec * pricing.output_cost_per_token
    per_req = pricing.per_request_cost
    total_cost = input_cost + output_cost + per_req

    return RequestCost(
        model=model,
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        total_tokens=total,
        input_cost=input_cost,
        output_cost=output_cost,
        per_request_cost=per_req,
        request_cost=total_cost,
        currency=currency,
        is_estimated=pricing.is_tiered,
    )


class ModelPricingCache:
    """
    Thread-safe and async-safe cache for AcademicAI model pricing.
    Single Source of Truth (SSOT) for model prices and cost currency.
    """

    def __init__(
        self,
        cache_file: Optional[Union[str, Path]] = None,
        ttl_seconds: Optional[int] = None,
        currency: Optional[str] = None,
    ):
        self._lock = threading.RLock()
        self._pricing_map: dict[str, ModelPricing] = {}
        self._last_refreshed_at: Optional[datetime] = None
        self._refresh_in_flight: bool = False
        self._explicit_file = cache_file
        self._explicit_ttl = ttl_seconds
        self._currency: Optional[str] = currency

        self._load_from_disk()

    @property
    def currency(self) -> str:
        with self._lock:
            if self._currency:
                return self._currency
            return str(_get_setting("COST_CURRENCY", "EUR"))

    @currency.setter
    def currency(self, val: str) -> None:
        with self._lock:
            self._currency = val

    @property
    def cache_file_path(self) -> Path:
        if self._explicit_file is not None:
            return Path(self._explicit_file)
        val = _get_setting("MODEL_PRICING_CACHE_FILE", "data/model_pricing_cache.json")
        return Path(val)

    @property
    def ttl_seconds(self) -> int:
        if self._explicit_ttl is not None:
            return self._explicit_ttl
        return int(_get_setting("MODEL_PRICING_CACHE_TTL_SECONDS", 86400))

    def _load_from_disk(self) -> None:
        p = self.cache_file_path
        if not p.exists():
            return
        with self._lock:
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return
                if "currency" in raw and raw["currency"]:
                    self._currency = str(raw["currency"])
                updated_str = raw.get("updated_at")
                if updated_str:
                    try:
                        self._last_refreshed_at = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
                    except Exception:
                        pass
                models_dict = raw.get("models")
                if isinstance(models_dict, dict):
                    for k, v in models_dict.items():
                        if isinstance(v, dict):
                            self._pricing_map[k] = ModelPricing.from_dict(v, default_currency=self._currency)
            except Exception as e:
                log.warning(f"failed to load model pricing cache from disk: {e}")

    def _save_to_disk(self) -> None:
        p = self.cache_file_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        with self._lock:
            payload = {
                "updated_at": self._last_refreshed_at.isoformat() if self._last_refreshed_at else datetime.now(timezone.utc).isoformat(),
                "ttl_seconds": self.ttl_seconds,
                "currency": self.currency,
                "models": {k: v.to_dict() for k, v in self._pricing_map.items()},
            }

            temp_file = tempfile.NamedTemporaryFile(
                mode="w",
                dir=p.parent,
                delete=False,
                encoding="utf-8",
                prefix=f".{p.name}.tmp-",
            )
            temp_path = Path(temp_file.name)
            closed = False
            try:
                temp_file.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                temp_file.flush()
                os.fsync(temp_file.fileno())
                temp_file.close()
                closed = True

                for attempt in range(5):
                    try:
                        os.replace(temp_path, p)
                        break
                    except PermissionError:
                        if attempt == 4:
                            raise
                        import time
                        time.sleep(0.02)
            except Exception as e:
                if not closed:
                    try:
                        temp_file.close()
                    except Exception:
                        pass
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                log.warning(f"failed to save model pricing cache to disk: {e}")

    def is_stale(self) -> bool:
        with self._lock:
            if self._last_refreshed_at is None:
                return True
            age = (datetime.now(timezone.utc) - self._last_refreshed_at).total_seconds()
            return age > self.ttl_seconds

    def get_pricing(self, model_id: str) -> Optional[ModelPricing]:
        with self._lock:
            return self._pricing_map.get(model_id)

    def get_pricing_with_refresh(self, model_id: str) -> Optional[ModelPricing]:
        if self.is_stale() and not self._refresh_in_flight:
            try:
                loop = asyncio.get_running_loop()
                self.trigger_background_refresh()
            except RuntimeError:
                # Synchronous environment (e.g. unit test or worker thread)
                try:
                    self.refresh_sync()
                except Exception as e:
                    log.warning(f"synchronous pricing refresh failed, falling back to cache: {e}")

        return self.get_pricing(model_id)

    def trigger_background_refresh(self) -> None:
        with self._lock:
            if self._refresh_in_flight:
                return
            self._refresh_in_flight = True

        async def _bg():
            try:
                await run_in_threadpool(self.refresh_sync)
            except Exception as e:
                log.warning(f"background model pricing refresh failed: {e}")
            finally:
                with self._lock:
                    self._refresh_in_flight = False

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_bg())
        except RuntimeError:
            with self._lock:
                self._refresh_in_flight = False

    def _fetch_from_backend(self) -> dict[str, ModelPricing]:
        base_url = get_base_url().rstrip("/")
        url = f"{base_url}/api/v1/llm/models"
        headers = dict(get_headers() or {})
        headers.setdefault("Accept", "application/json")

        with httpx.Client(timeout=10.0, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        models_list: list[dict[str, Any]] = []
        if isinstance(payload, list):
            models_list = payload
        elif isinstance(payload, dict):
            if isinstance(payload.get("data"), list):
                models_list = payload["data"]
            elif isinstance(payload.get("models"), list):
                models_list = payload["models"]

        fresh_map: dict[str, ModelPricing] = {}
        for m in models_list:
            if not isinstance(m, dict):
                continue
            m_id = m.get("modelName") or m.get("id") or m.get("name")
            if not m_id:
                continue
            raw_costs = m.get("costs") or []
            if isinstance(raw_costs, list):
                fresh_map[str(m_id)] = parse_model_costs(str(m_id), raw_costs)

        return fresh_map

    def refresh_sync(self) -> dict[str, ModelPricing]:
        with self._lock:
            fresh_map = self._fetch_from_backend()
            if fresh_map:
                self._pricing_map.update(fresh_map)
                self._last_refreshed_at = datetime.now(timezone.utc)
                self._save_to_disk()
            return dict(self._pricing_map)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "models_cached": len(self._pricing_map),
                "last_refreshed_at": self._last_refreshed_at.isoformat() if self._last_refreshed_at else None,
                "is_stale": self.is_stale(),
                "ttl_seconds": self.ttl_seconds,
                "currency": self.currency,
                "cache_file": str(self.cache_file_path),
            }


_pricing_cache: Optional[ModelPricingCache] = None
_pricing_cache_lock = threading.RLock()


def get_pricing_cache() -> ModelPricingCache:
    global _pricing_cache
    with _pricing_cache_lock:
        if _pricing_cache is None:
            _pricing_cache = ModelPricingCache()
        return _pricing_cache
