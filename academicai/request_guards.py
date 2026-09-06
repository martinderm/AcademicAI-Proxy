"""
AcademicAI Request Guards — Inbound HTTP validation and rate limiting.

Handles:
- Request body JSON serialization and size checks (413/422)
- OpenAI-compatible chat completion payload validation (422)
- Threshold validation for messages, content lengths, tools and schemas (413)
- In-memory token bucket rate limiting with periodic TTL sweep cleanup (429)
"""

import json
import logging
import sys
import threading
import time
from typing import Any, Optional

from fastapi import HTTPException, Request

import academicai.config as config

log = logging.getLogger("proxy")

_rate_limit_lock = threading.Lock()
_rate_limit_buckets: dict[str, list[float]] = {}
_last_rate_limit_sweep: float = 0.0
DEFAULT_SWEEP_INTERVAL_SECONDS: float = 60.0


from academicai.transformation import extract_text_content, _extract_text_content


def _get_limit(name: str, explicit_value: Optional[Any] = None) -> Any:
    """
    Liefert das Konfigurationslimit.
    Priorität:
    1. Explizit übergebener Parameter (falls ungleich None)
    2. Dynamisch gesetztes Attribut auf server (für monkeypatch in Tests)
    3. Modul academicai.config
    """
    if explicit_value is not None:
        return explicit_value

    server_mod = sys.modules.get("server")
    if server_mod is not None and hasattr(server_mod, name):
        return getattr(server_mod, name)

    return getattr(config, name, None)


def validate_request_json_size(body: Any, max_chars: Optional[int] = None) -> int:
    """
    Prüft, ob der Request-Body JSON-serialisierbar ist und die maximale Gesamtgröße einhält.

    Raises:
        HTTPException(422): Wenn body nicht JSON-serialisierbar ist.
        HTTPException(413): Wenn serialisierte Größe max_chars übersteigt.
    """
    limit = int(_get_limit("MAX_REQUEST_JSON_CHARS", max_chars))
    try:
        request_size = len(json.dumps(body, ensure_ascii=False))
    except Exception:
        log.warning("Chat request rejected (422): request body is not JSON-serializable")
        raise HTTPException(status_code=422, detail="request body is not JSON-serializable")

    if request_size > limit:
        log.warning(
            f"Chat request rejected (413): request body size {request_size} chars "
            f"exceeds limit ({limit} chars)"
        )
        raise HTTPException(status_code=413, detail=f"request body exceeds limit ({limit} chars)")

    return request_size


_validate_request_json_size = validate_request_json_size


def validate_chat_request_body(
    body: dict,
    max_messages: Optional[int] = None,
    max_message_text_chars: Optional[int] = None,
    max_tools: Optional[int] = None,
    max_tool_schema_chars: Optional[int] = None,
) -> None:
    """
    Validiert den Inbound-Chat-Request-Body gegen OpenAI-Schemas und Schutzgrenzen.

    Raises:
        HTTPException(422): Bei ungültigen Schema-Strukturen, ungültigen Typen oder fehlenden Pflichtfeldern.
        HTTPException(413): Wenn Nachrichtenanzahl, Textlänge, Tool-Anzahl oder Schemagröße Schutzgrenzen übersteigen.
    """
    if not isinstance(body, dict):
        log.warning("Validation rejected (422): request body must be a JSON object")
        raise HTTPException(status_code=422, detail="request body must be a JSON object")

    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        log.warning("Validation rejected (422): model must be a non-empty string")
        raise HTTPException(status_code=422, detail="model must be a non-empty string")
    if len(model.strip()) > 200:
        log.warning(f"Validation rejected (422): model name too long ({len(model.strip())} chars > 200)")
        raise HTTPException(status_code=422, detail="model is too long")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        log.warning("Validation rejected (422): messages must be a non-empty list")
        raise HTTPException(status_code=422, detail="messages must be a non-empty list")

    limit_messages = int(_get_limit("MAX_MESSAGES", max_messages))
    if len(messages) > limit_messages:
        log.warning(f"Validation rejected (413): messages count {len(messages)} exceeds limit ({limit_messages})")
        raise HTTPException(status_code=413, detail=f"messages exceed limit ({limit_messages})")

    limit_msg_text = int(_get_limit("MAX_MESSAGE_TEXT_CHARS", max_message_text_chars))
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            log.warning(f"Validation rejected (422): messages[{idx}] must be an object")
            raise HTTPException(status_code=422, detail=f"messages[{idx}] must be an object")
        role = msg.get("role")
        if not isinstance(role, str) or not role.strip():
            log.warning(f"Validation rejected (422): messages[{idx}].role must be a non-empty string")
            raise HTTPException(status_code=422, detail=f"messages[{idx}].role must be a non-empty string")

        content_text = extract_text_content(msg.get("content"))
        if len(content_text) > limit_msg_text:
            log.warning(
                f"Validation rejected (413): messages[{idx}].content length {len(content_text)} chars "
                f"exceeds limit ({limit_msg_text} chars)"
            )
            raise HTTPException(
                status_code=413,
                detail=f"messages[{idx}].content exceeds limit ({limit_msg_text} chars)",
            )

    tools = body.get("tools") or body.get("functions") or []
    if tools and not isinstance(tools, list):
        log.warning("Validation rejected (422): tools/functions must be a list")
        raise HTTPException(status_code=422, detail="tools/functions must be a list")

    limit_tools = int(_get_limit("MAX_TOOLS", max_tools))
    if isinstance(tools, list) and len(tools) > limit_tools:
        log.warning(f"Validation rejected (413): tools count {len(tools)} exceeds limit ({limit_tools})")
        raise HTTPException(status_code=413, detail=f"tools exceed limit ({limit_tools})")

    limit_tool_schema = int(_get_limit("MAX_TOOL_SCHEMA_CHARS", max_tool_schema_chars))
    for idx, tool in enumerate(tools or []):
        if not isinstance(tool, dict):
            log.warning(f"Validation rejected (422): tools[{idx}] must be an object")
            raise HTTPException(status_code=422, detail=f"tools[{idx}] must be an object")
        try:
            schema_size = len(json.dumps(tool, ensure_ascii=False))
        except Exception:
            log.warning(f"Validation rejected (422): tools[{idx}] is not JSON-serializable")
            raise HTTPException(status_code=422, detail=f"tools[{idx}] is not JSON-serializable")
        if schema_size > limit_tool_schema:
            fn = tool.get("function")
            tool_name = (fn.get("name") if isinstance(fn, dict) else None) or f"index_{idx}"
            log.warning(
                f"Validation rejected (413): tool '{tool_name}' schema size {schema_size} chars "
                f"exceeds limit ({limit_tool_schema} chars)"
            )
            raise HTTPException(
                status_code=413,
                detail=f"tools[{idx}] exceeds limit ({limit_tool_schema} chars)",
            )


_validate_chat_request_body = validate_chat_request_body


def _rate_limit_bucket(request: Request, key: str) -> str:
    """Ermittelt den Rate-Limit-Bucket-Bezeichner aus Client-Host und Token-Präfix."""
    client_host = request.client.host if (request and hasattr(request, "client") and request.client) else "unknown"
    key_str = str(key or "")
    return f"{client_host}:{key_str[:8]}"


rate_limit_bucket = _rate_limit_bucket


def _prune_rate_limit_buckets_locked(now: float, window_seconds: float) -> int:
    """
    Pruning-Routine bei bereits gehaltenem _rate_limit_lock.
    Entfernt leere/veraltete Buckets und schneidet abgelaufene Hits ab.
    """
    window_start = now - float(window_seconds)
    stale_keys = []
    for key, hits in list(_rate_limit_buckets.items()):
        active = [ts for ts in hits if ts >= window_start]
        if not active:
            stale_keys.append(key)
        elif len(active) < len(hits):
            _rate_limit_buckets[key] = active
    for key in stale_keys:
        del _rate_limit_buckets[key]
    return len(stale_keys)


def prune_rate_limit_buckets(
    now: Optional[float] = None,
    window_seconds: Optional[float] = None,
) -> int:
    """
    Bereinigt _rate_limit_buckets: entfernt abgelaufene Zeitstempel und prunt verwaiste/leere Buckets.

    Args:
        now: Optionaler Zeitstempel (Standard: time.time()).
        window_seconds: Optionales Zeitfenster in Sekunden (Standard: RATE_LIMIT_WINDOW_SECONDS).

    Returns:
        Anzahl der vollständig gelöschten Buckets.
    """
    if now is None:
        now = time.time()
    if window_seconds is None:
        window_seconds = float(_get_limit("RATE_LIMIT_WINDOW_SECONDS", 60.0))

    with _rate_limit_lock:
        return _prune_rate_limit_buckets_locked(now, window_seconds)


_prune_rate_limit_buckets = prune_rate_limit_buckets


def _enforce_chat_rate_limit(
    request: Request,
    key: str,
    rate_limit_per_minute: Optional[int] = None,
    rate_limit_window_seconds: Optional[int] = None,
    sweep_interval_seconds: Optional[float] = None,
) -> None:
    """
    Erzwingt In-Memory Token-Bucket Rate Limiting pro Client-Host + Token.
    Führt periodisch einen Sweep durch, um unbegrenztes Speicherwachstum zu verhindern.

    Raises:
        HTTPException(429): Wenn das Rate Limit überschritten wird.
    """
    limit = int(_get_limit("RATE_LIMIT_PER_MINUTE", rate_limit_per_minute))
    if limit <= 0:
        return

    window = float(_get_limit("RATE_LIMIT_WINDOW_SECONDS", rate_limit_window_seconds))
    sweep_interval = float(
        sweep_interval_seconds
        if sweep_interval_seconds is not None
        else _get_limit("RATE_LIMIT_SWEEP_INTERVAL_SECONDS", DEFAULT_SWEEP_INTERVAL_SECONDS)
    )

    bucket = _rate_limit_bucket(request, key)
    now = time.time()
    window_start = now - window

    global _last_rate_limit_sweep

    with _rate_limit_lock:
        # Periodischer Sweep / TTL-Cleanup verwaister Buckets
        if now - _last_rate_limit_sweep >= sweep_interval:
            _prune_rate_limit_buckets_locked(now, window)
            _last_rate_limit_sweep = now

        hits = [ts for ts in _rate_limit_buckets.get(bucket, []) if ts >= window_start]
        if len(hits) >= limit:
            raise HTTPException(status_code=429, detail="rate limit exceeded")
        hits.append(now)
        _rate_limit_buckets[bucket] = hits


enforce_chat_rate_limit = _enforce_chat_rate_limit

__all__ = [
    "extract_text_content",
    "_extract_text_content",
    "validate_request_json_size",
    "_validate_request_json_size",
    "validate_chat_request_body",
    "_validate_chat_request_body",
    "rate_limit_bucket",
    "_rate_limit_bucket",
    "prune_rate_limit_buckets",
    "_prune_rate_limit_buckets",
    "enforce_chat_rate_limit",
    "_enforce_chat_rate_limit",
    "_rate_limit_buckets",
    "_rate_limit_lock",
]
