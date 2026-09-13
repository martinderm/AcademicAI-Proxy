"""
AcademicAI Local Cost Tracker — Aggregations, Ring-Buffer Request History & Atomic Persistence.

Provides:
- CostAggregationBucket: numerical accumulator for tokens, costs, and request counts
- LocalCostStore: thread-safe persistent store with all-time, daily, monthly, per-model, and per-client tracking
- Ring buffer of recent request metadata (strictly privacy-compliant: NO prompts, completions, or keys)
- Atomic JSON persistence on disk
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Union

from academicai.cost_calculation import RequestCost
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


def hash_client_key(key: Optional[str]) -> str:
    """Returns an anonymized client identifier hash (client_<sha256[:8]>)."""
    if not key or not str(key).strip():
        return "client_anonymous"
    clean_key = str(key).strip()
    h = hashlib.sha256(clean_key.encode("utf-8")).hexdigest()[:8]
    return f"client_{h}"


@dataclass
class CostAggregationBucket:
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    input_cost: Decimal = Decimal("0")
    output_cost: Decimal = Decimal("0")
    per_request_cost: Decimal = Decimal("0")
    request_cost: Decimal = Decimal("0")

    def add(self, cost: RequestCost) -> None:
        self.request_count += 1
        self.prompt_tokens += cost.prompt_tokens
        self.completion_tokens += cost.completion_tokens
        self.total_tokens += cost.total_tokens
        self.input_cost += cost.input_cost
        self.output_cost += cost.output_cost
        self.per_request_cost += cost.per_request_cost
        self.request_cost += cost.request_cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": float(self.input_cost),
            "output_cost": float(self.output_cost),
            "per_request_cost": float(self.per_request_cost),
            "request_cost": float(self.request_cost),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostAggregationBucket":
        return cls(
            request_count=int(data.get("request_count", 0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            input_cost=Decimal(str(data.get("input_cost", "0"))),
            output_cost=Decimal(str(data.get("output_cost", "0"))),
            per_request_cost=Decimal(str(data.get("per_request_cost", "0"))),
            request_cost=Decimal(str(data.get("request_cost", "0"))),
        )


class LocalCostStore:
    """
    Thread-safe and persistent local store for cost tracking.
    """

    def __init__(
        self,
        cache_file: Optional[Union[str, Path]] = None,
        history_limit: Optional[int] = None,
        enabled: Optional[bool] = None,
    ):
        self._lock = threading.RLock()
        self._explicit_file = cache_file
        self._explicit_history_limit = history_limit
        self._explicit_enabled = enabled

        limit = self.history_limit
        self._recent_requests: deque[dict[str, Any]] = deque(maxlen=limit)
        self._all_time = CostAggregationBucket()
        self._today: dict[str, CostAggregationBucket] = {}
        self._this_month: dict[str, CostAggregationBucket] = {}
        self._by_model: dict[str, CostAggregationBucket] = {}
        self._by_client: dict[str, CostAggregationBucket] = {}

        self._load_from_disk()

    @property
    def cache_file_path(self) -> Path:
        if self._explicit_file is not None:
            return Path(self._explicit_file)
        val = _get_setting("LOCAL_COST_CACHE_FILE", "data/local_cost_cache.json")
        return Path(val)

    @property
    def history_limit(self) -> int:
        if self._explicit_history_limit is not None:
            return self._explicit_history_limit
        return int(_get_setting("LOCAL_COST_HISTORY_LIMIT", 500))

    @property
    def is_enabled(self) -> bool:
        if self._explicit_enabled is not None:
            return self._explicit_enabled
        return bool(_get_setting("ENABLE_LOCAL_COST_TRACKING", True))

    def _load_from_disk(self) -> None:
        p = self.cache_file_path
        if not p.exists():
            return
        with self._lock:
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return
                if "all_time" in raw and isinstance(raw["all_time"], dict):
                    self._all_time = CostAggregationBucket.from_dict(raw["all_time"])
                if "today" in raw and isinstance(raw["today"], dict):
                    self._today = {k: CostAggregationBucket.from_dict(v) for k, v in raw["today"].items()}
                if "this_month" in raw and isinstance(raw["this_month"], dict):
                    self._this_month = {k: CostAggregationBucket.from_dict(v) for k, v in raw["this_month"].items()}
                if "by_model" in raw and isinstance(raw["by_model"], dict):
                    self._by_model = {k: CostAggregationBucket.from_dict(v) for k, v in raw["by_model"].items()}
                if "by_client" in raw and isinstance(raw["by_client"], dict):
                    self._by_client = {k: CostAggregationBucket.from_dict(v) for k, v in raw["by_client"].items()}
                if "recent_requests" in raw and isinstance(raw["recent_requests"], list):
                    self._recent_requests = deque(raw["recent_requests"], maxlen=self.history_limit)
            except Exception as e:
                log.warning(f"failed to load local cost cache from disk: {e}")

    def _save_to_disk(self) -> None:
        p = self.cache_file_path
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        with self._lock:
            payload = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "all_time": self._all_time.to_dict(),
                "today": {k: v.to_dict() for k, v in self._today.items()},
                "this_month": {k: v.to_dict() for k, v in self._this_month.items()},
                "by_model": {k: v.to_dict() for k, v in self._by_model.items()},
                "by_client": {k: v.to_dict() for k, v in self._by_client.items()},
                "recent_requests": list(self._recent_requests),
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
                log.warning(f"failed to save local cost cache: {e}")

    def record_request(self, cost: RequestCost, client_key: Optional[str] = None) -> None:
        """
        Records a completed request's cost and tokens exactly once.
        Strict privacy: no prompt or completion text is ever stored.
        """
        if not self.is_enabled:
            return

        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        month_str = now_utc.strftime("%Y-%m")
        client_id = hash_client_key(client_key)

        with self._lock:
            # All-time
            self._all_time.add(cost)

            # Today
            if today_str not in self._today:
                self._today[today_str] = CostAggregationBucket()
            self._today[today_str].add(cost)

            # This month
            if month_str not in self._this_month:
                self._this_month[month_str] = CostAggregationBucket()
            self._this_month[month_str].add(cost)

            # By model
            if cost.model not in self._by_model:
                self._by_model[cost.model] = CostAggregationBucket()
            self._by_model[cost.model].add(cost)

            # By client
            if client_id not in self._by_client:
                self._by_client[client_id] = CostAggregationBucket()
            self._by_client[client_id].add(cost)

            # Recent requests ring buffer
            entry = {
                "timestamp": now_utc.isoformat(),
                "model": cost.model,
                "client_id": client_id,
                "prompt_tokens": cost.prompt_tokens,
                "completion_tokens": cost.completion_tokens,
                "total_tokens": cost.total_tokens,
                "request_cost": float(cost.request_cost),
                "currency": cost.currency,
                "is_estimated": cost.is_estimated,
            }
            self._recent_requests.append(entry)

            self._save_to_disk()

    def get_status_payload(self) -> dict[str, Any]:
        with self._lock:
            now_utc = datetime.now(timezone.utc)
            today_str = now_utc.strftime("%Y-%m-%d")
            month_str = now_utc.strftime("%Y-%m")

            today_bucket = self._today.get(today_str, CostAggregationBucket())
            month_bucket = self._this_month.get(month_str, CostAggregationBucket())

            return {
                "enabled": self.is_enabled,
                "all_time": self._all_time.to_dict(),
                "today": today_bucket.to_dict(),
                "this_month": month_bucket.to_dict(),
                "by_model": {k: v.to_dict() for k, v in self._by_model.items()},
                "by_client": {k: v.to_dict() for k, v in self._by_client.items()},
                "recent_requests_count": len(self._recent_requests),
                "recent_requests": list(self._recent_requests),
            }


_store_instance: Optional[LocalCostStore] = None
_store_lock = threading.RLock()


def get_local_cost_store() -> LocalCostStore:
    global _store_instance
    with _store_lock:
        if _store_instance is None:
            _store_instance = LocalCostStore()
        return _store_instance
