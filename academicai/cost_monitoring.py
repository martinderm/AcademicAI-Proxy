"""
AcademicAI Cost Monitoring — Cache, API-Refresh und Header-Generierung.

Kapselt:
- Parsen von Kosten-Snapshots der BOKU-API
- Lokales Datei-Caching mit atomarem Schreiben und Verzeichnis-Erstellung
- Stale-Erkennung und asynchrones / synchrones Cache-Refresh
- Injektion von AcademicAI-Cost-Headern in HTTP-Responses
- Status-Payload für /internal/cost-status
- Dynamische Konfigurationsauflösung über server / academicai.config
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import httpx
from starlette.concurrency import run_in_threadpool

from academicai.auth import get_base_url, get_headers
import academicai.config as config

log = logging.getLogger("proxy")

_cost_lock = threading.RLock()
_cost_refresh_in_flight = False


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_setting(name: str, explicit_value: Optional[Any] = None) -> Any:
    """
    Liefert eine Konfigurationseinstellung oder ein Hook-Callable.
    Priorität:
    1. Explizit übergebener Parameter (falls ungleich None)
    2. Dynamisch gesetztes Attribut auf server (z.B. monkeypatch in Tests)
    3. Modul academicai.config (falls vorhanden)
    """
    if explicit_value is not None:
        return explicit_value

    server_mod = sys.modules.get("server")
    if server_mod is not None and hasattr(server_mod, name):
        return getattr(server_mod, name)

    if hasattr(config, name):
        return getattr(config, name)

    return None


def _parse_iso_ts(raw: Any) -> Optional[datetime]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        value = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


parse_iso_ts = _parse_iso_ts


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


safe_float = _safe_float


def _extract_cost_summary(payload: Any) -> dict[str, Any]:
    root = payload if isinstance(payload, dict) else {}
    data = root.get("data") if isinstance(root.get("data"), dict) else root

    total_cost = _safe_float(data.get("totalCost"))
    total_clients = data.get("totalClients")
    costs = data.get("costs") if isinstance(data.get("costs"), list) else []

    try:
        total_clients = int(total_clients) if total_clients is not None else None
    except Exception:
        total_clients = None

    return {
        "total_cost": total_cost,
        "total_clients": total_clients,
        "cost_entries": len(costs),
    }


extract_cost_summary = _extract_cost_summary


def read_cost_cache(cache_file: Optional[Union[str, Path]] = None) -> dict[str, Any]:
    file_path = _get_setting("COST_CACHE_FILE", cache_file)
    p = Path(file_path)
    if not p.exists():
        return {}
    with _cost_lock:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


_read_cost_cache = read_cost_cache


def write_cost_cache(
    cache: dict[str, Any],
    cache_file: Optional[Union[str, Path]] = None,
) -> None:
    file_path = _get_setting("COST_CACHE_FILE", cache_file)
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with _cost_lock:
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
            temp_file.write(json.dumps(cache, ensure_ascii=False, indent=2) + "\n")
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file.close()
            closed = True

            # Windows retry on sharing violation / transient file lock
            for attempt in range(5):
                try:
                    os.replace(temp_path, p)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    import time
                    time.sleep(0.02)
        except Exception:
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
            raise



_write_cost_cache = write_cost_cache


def is_cost_cache_stale(
    cache: Any,
    ttl_seconds: Optional[Union[int, float]] = None,
) -> bool:
    if not isinstance(cache, dict):
        return True
    ts = _parse_iso_ts(str(cache.get("updated_at", "")))
    if ts is None:
        return True
    ttl = _get_setting("COST_CACHE_TTL_SECONDS", ttl_seconds)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() > ttl


_is_cost_cache_stale = is_cost_cache_stale


def build_cost_headers(
    cache: Any,
    enabled: Optional[bool] = None,
) -> dict[str, str]:
    is_enabled = bool(_get_setting("ENABLE_COST_MONITORING", enabled))
    if not is_enabled:
        return {}
    if not cache or not isinstance(cache, dict):
        return {}

    stale_fn = _get_setting("_is_cost_cache_stale", is_cost_cache_stale)
    safe_float_fn = _get_setting("_safe_float", _safe_float)

    headers: dict[str, str] = {
        "X-AcademicAI-Cost-Stale": "true" if stale_fn(cache) else "false",
    }
    updated_at = str(cache.get("updated_at", "")).strip()
    if updated_at:
        headers["X-AcademicAI-Cost-Updated-At"] = updated_at

    total_cost = safe_float_fn(cache.get("total_cost"))
    if total_cost is not None:
        headers["X-AcademicAI-Total-Cost"] = f"{total_cost:.6f}".rstrip("0").rstrip(".")

    total_clients = cache.get("total_clients")
    if isinstance(total_clients, int):
        headers["X-AcademicAI-Total-Clients"] = str(total_clients)

    cost_entries = cache.get("cost_entries")
    if isinstance(cost_entries, int):
        headers["X-AcademicAI-Cost-Entries"] = str(cost_entries)

    return headers


_build_cost_headers = build_cost_headers


def fetch_cost_snapshot(
    enabled: Optional[bool] = None,
    timeout_seconds: Optional[float] = None,
) -> dict[str, Any]:
    if not _get_setting("ENABLE_COST_MONITORING", enabled):
        return {}

    base_url = get_base_url().rstrip("/")
    cost_url = f"{base_url}/api/v1/cost/"
    headers = dict(get_headers() or {})
    headers.setdefault("Accept", "application/json")

    timeout = _get_setting("COST_REFRESH_TIMEOUT_SECONDS", timeout_seconds)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(cost_url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()

    summary_fn = _get_setting("_extract_cost_summary", _extract_cost_summary)
    summary = summary_fn(payload)
    return {
        "updated_at": _now_utc_iso(),
        "source": "live",
        "raw": payload,
        **summary,
    }


_fetch_cost_snapshot = fetch_cost_snapshot


def refresh_cost_cache_sync(
    enabled: Optional[bool] = None,
    cache_file: Optional[Union[str, Path]] = None,
    timeout_seconds: Optional[float] = None,
) -> dict[str, Any]:
    if not _get_setting("ENABLE_COST_MONITORING", enabled):
        read_fn = _get_setting("_read_cost_cache", read_cost_cache)
        return read_fn(cache_file=cache_file)

    fetch_fn = _get_setting("_fetch_cost_snapshot", fetch_cost_snapshot)
    write_fn = _get_setting("_write_cost_cache", write_cost_cache)

    with _cost_lock:
        fresh = fetch_fn(enabled=enabled, timeout_seconds=timeout_seconds)
        if fresh:
            write_fn(fresh, cache_file=cache_file)
        return fresh


_refresh_cost_cache_sync = refresh_cost_cache_sync


async def refresh_cost_cache_background(
    enabled: Optional[bool] = None,
    cache_file: Optional[Union[str, Path]] = None,
    timeout_seconds: Optional[float] = None,
) -> None:
    global _cost_refresh_in_flight
    try:
        sync_fn = _get_setting("_refresh_cost_cache_sync", refresh_cost_cache_sync)
        try:
            await run_in_threadpool(
                sync_fn,
                enabled=enabled,
                cache_file=cache_file,
                timeout_seconds=timeout_seconds,
            )
        except TypeError:
            await run_in_threadpool(sync_fn)
    except Exception as e:
        log.warning(f"cost refresh failed: {e}")
    finally:
        _cost_refresh_in_flight = False


_refresh_cost_cache_background = refresh_cost_cache_background


def get_cost_cache_with_lazy_refresh(
    enabled: Optional[bool] = None,
    cache_file: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    global _cost_refresh_in_flight
    read_fn = _get_setting("_read_cost_cache", read_cost_cache)
    cache = read_fn(cache_file=cache_file)

    if not _get_setting("ENABLE_COST_MONITORING", enabled):
        return cache

    stale_fn = _get_setting("_is_cost_cache_stale", is_cost_cache_stale)
    if stale_fn(cache) and not _cost_refresh_in_flight:
        try:
            loop = asyncio.get_running_loop()
            _cost_refresh_in_flight = True
            refresh_bg_fn = _get_setting(
                "_refresh_cost_cache_background", refresh_cost_cache_background
            )
            loop.create_task(
                refresh_bg_fn(enabled=enabled, cache_file=cache_file)
            )
        except RuntimeError:
            # Kein laufender Loop (z.B. in unit tests) -> synchron vermeiden
            pass

    return cache


_get_cost_cache_with_lazy_refresh = get_cost_cache_with_lazy_refresh


def get_cost_status_payload(
    cache: Optional[dict[str, Any]] = None,
    enabled: Optional[bool] = None,
) -> dict[str, Any]:
    if enabled is None:
        enabled = bool(_get_setting("ENABLE_COST_MONITORING"))
    if cache is None:
        get_cache_fn = _get_setting(
            "_get_cost_cache_with_lazy_refresh", get_cost_cache_with_lazy_refresh
        )
        cache = get_cache_fn(enabled=enabled)

    safe_float_fn = _get_setting("_safe_float", _safe_float)
    stale_fn = _get_setting("_is_cost_cache_stale", is_cost_cache_stale)

    return {
        "enabled": enabled,
        "total_cost": safe_float_fn(cache.get("total_cost")),
        "total_clients": cache.get("total_clients"),
        "cost_entries": cache.get("cost_entries"),
        "updated_at": cache.get("updated_at"),
        "is_stale": stale_fn(cache) if cache else True,
        "source": cache.get("source", "cache" if cache else "none"),
    }
