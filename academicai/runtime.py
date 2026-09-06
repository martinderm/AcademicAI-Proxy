"""
AcademicAI Runtime Lifecycle Helpers — PID-File Management und Backend Health Checks.

Kapselt:
- Schreiben und Bereinigen von PID-Dateien mit Validierung
- Backend-Connectivity Health Checks gegen BOKU /api/v1/llm/models mit Latenzmessung
- Erstellung standardisierter Health-Payloads für /health
- Dynamische Konfigurationsauflösung über server / academicai.config
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

import httpx

from academicai.auth import get_base_url, get_headers
import academicai.config as config

log = logging.getLogger("academicai-proxy")


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


def write_pid_file(pid_file: Optional[Union[str, Path]] = None) -> None:
    """
    Schreibt die PID des aktuellen Prozesses in die PID-Datei.
    Stellt sicher, dass das übergeordnete Verzeichnis existiert.
    """
    target = _get_setting("PID_FILE", pid_file)
    try:
        if target is None:
            target = Path("server.pid")
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    except Exception as e:
        log.warning(f"could not write pid file {target}: {e}")


_write_pid_file = write_pid_file


def cleanup_pid_file(pid_file: Optional[Union[str, Path]] = None) -> None:
    """
    Entfernt die PID-Datei, jedoch nur wenn deren Inhalt mit der aktuellen PID übereinstimmt.
    """
    target = _get_setting("PID_FILE", pid_file)
    try:
        if target is None:
            return
        path = Path(target)
        if not path.exists():
            return
        raw = path.read_text(encoding="utf-8").strip()
        if raw and raw != str(os.getpid()):
            return
        path.unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"could not cleanup pid file {target}: {e}")


_cleanup_pid_file = cleanup_pid_file


def check_backend_health(
    enabled: Optional[bool] = None,
    timeout: Optional[float] = None,
    *,
    base_url: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
    client: Optional[httpx.Client] = None,
) -> dict[str, Any]:
    """
    Führt einen Health Check gegen den BOKU-Endpunkt /api/v1/llm/models durch.
    Misst die Latenz und meldet Status bzw. Fehlermeldungen.
    """
    is_enabled = _get_setting("HEALTH_CHECK_BACKEND", enabled)
    if is_enabled is None:
        is_enabled = True
    if not is_enabled:
        return {"enabled": False, "ok": None}

    timeout_val = _get_setting("HEALTH_CHECK_TIMEOUT_SECONDS", timeout)
    if timeout_val is None:
        timeout_seconds = 2.0
    else:
        try:
            timeout_seconds = float(timeout_val)
        except Exception:
            timeout_seconds = 2.0

    b_url = base_url
    if b_url is None:
        base_url_getter = _get_setting("get_base_url", None)
        if callable(base_url_getter):
            b_url = base_url_getter()
        else:
            b_url = get_base_url()

    hdrs = headers
    if hdrs is None:
        headers_getter = _get_setting("get_headers", None)
        if callable(headers_getter):
            hdrs = headers_getter()
        else:
            hdrs = get_headers()
    hdrs = dict(hdrs or {})

    endpoint = f"{str(b_url).rstrip('/')}/api/v1/llm/models"
    started = time.perf_counter()
    try:
        if client is not None:
            resp = client.get(endpoint, headers=hdrs)
        else:
            with httpx.Client(timeout=timeout_seconds) as c:
                resp = c.get(endpoint, headers=hdrs)

        latency_ms = int((time.perf_counter() - started) * 1000)
        ok = resp.status_code == 200
        out: dict[str, Any] = {
            "enabled": True,
            "ok": ok,
            "status_code": resp.status_code,
            "latency_ms": latency_ms,
        }
        if not ok:
            out["error"] = "backend responded with non-200 status"
        return out
    except Exception as e:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "enabled": True,
            "ok": False,
            "latency_ms": latency_ms,
            "error": str(e),
        }


_check_backend_health = check_backend_health


def get_health_payload(backend: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Erstellt das standardisierte JSON-Payload für /health:
    {"status": "ok"|"degraded", "service": "academicai-proxy", "backend": ...}
    """
    if backend is None:
        backend = check_backend_health()

    status = "ok"
    if backend.get("enabled") and backend.get("ok") is False:
        status = "degraded"

    return {
        "status": status,
        "service": "academicai-proxy",
        "backend": backend,
    }


__all__ = [
    "write_pid_file",
    "_write_pid_file",
    "cleanup_pid_file",
    "_cleanup_pid_file",
    "check_backend_health",
    "_check_backend_health",
    "get_health_payload",
]
