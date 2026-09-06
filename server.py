"""
AcademicAI OpenAI-kompatibler Proxy Server

Exponiert AcademicAI als OpenAI-kompatible API auf Port 11435.
OpenClaw (und andere Tools) können ihn wie jeden OpenAI-kompatiblen Provider nutzen.

Start:
    py server.py

Endpoints:
    GET  /v1/models                 → Modell-Liste
    POST /v1/chat/completions       → Chat Completion (inkl. Streaming-Emulation)
    GET  /health                    → Health Check
"""

import os
import sys
import time
import uuid
import json
import logging
import re
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool
import uvicorn

import academicai
from academicai.auth import get_base_url, get_headers
from academicai.security import redact_sensitive
from academicai.tool_emulation import (
    inject_tools_into_messages,
    parse_tool_calls,
    extract_respond_content,
    format_arbitrary_json_as_codeblock,
    format_arbitrary_json_for_humans,
    build_tool_calls_response,
    build_tool_calls_sse_chunks,
)
from academicai.request_guards import (
    extract_text_content,
    _extract_text_content,
    validate_request_json_size,
    _validate_request_json_size,
    validate_chat_request_body,
    _validate_chat_request_body,
    rate_limit_bucket,
    _rate_limit_bucket,
    prune_rate_limit_buckets,
    _prune_rate_limit_buckets,
    enforce_chat_rate_limit,
    _enforce_chat_rate_limit,
    _rate_limit_buckets,
    _rate_limit_lock,
)


def _last_user_text(messages: list) -> str:
    """Liefert den letzten User-Text aus den Original-Messages."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return _extract_text_content(m.get("content"))
    return ""


def _apply_post_tool_guard(messages: list, has_tools: bool) -> list:
    """
    Stabilisiert den Follow-up-Schritt nach einem Tool-Result.
    - Erfolgreiches Tool-Result: finale Antwort bevorzugen.
    - Fehlerhaftes Tool-Result: Erfolg NICHT behaupten, sondern korrigierten
      Tool-Call auslösen oder Fehler transparent melden.
    """
    if not has_tools or not messages:
        return messages

    last = messages[-1] or {}
    if last.get("role") != "tool":
        return messages

    tool_text = _extract_text_content(last.get("content")).lower()
    has_error = any(tok in tool_text for tok in ["error:", "cannot parse", "failed", "not found", "exception"])

    if has_error:
        guard_text = (
            "TOOL_RESULT_ERROR: The latest tool result contains an error. "
            "Do NOT claim success. Either issue a corrected tool_call, or explain the failure clearly. "
            "For mailbox envelope search, keep options before query, e.g. envelope list -s 50 \"from alerts@example.com\"."
        )
    else:
        guard_text = (
            "NO_FURTHER_TOOL_CALLS: You already received tool results. "
            "Now produce the final user-facing answer. "
            "Call another tool only if the latest tool result is clearly missing required data."
        )

    return [{"role": "system", "content": guard_text}] + messages


def _is_human_readable_target(messages: list) -> bool:
    """
    Heuristik: Nur bei menschlichen Zielkanälen JSON->Human-Text-Fallback aktivieren.

    False für klar maschinelle Runs (z.B. cron).
    True für typische Human-Channels (whatsapp/telegram/signal/discord/slack/webchat...).
    """
    user_text = "\n".join(
        _extract_text_content(m.get("content"))
        for m in messages
        if m.get("role") == "user"
    ).lower()

    # Explizit maschineller Trigger
    if "[cron:" in user_text:
        return False

    # Chat-Metadaten aus OpenClaw-User-Envelope (auch ohne system channel marker)
    user_human_markers = [
        "conversation info (untrusted metadata)",
        '"is_group_chat": true',
        '"is_group_chat": false',
        '"conversation_label":',
        '"sender": "+',
    ]
    if any(marker in user_text for marker in user_human_markers):
        return True

    system_text = "\n".join(
        _extract_text_content(m.get("content"))
        for m in messages
        if m.get("role") == "system"
    ).lower()

    human_channel_markers = [
        "channel=whatsapp", '"channel": "whatsapp"',
        "channel=telegram", '"channel": "telegram"',
        "channel=signal", '"channel": "signal"',
        "channel=imessage", '"channel": "imessage"',
        "channel=discord", '"channel": "discord"',
        "channel=slack", '"channel": "slack"',
        "channel=googlechat", '"channel": "googlechat"',
        "channel=irc", '"channel": "irc"',
        "channel=webchat", '"channel": "webchat"',
        '"chat_type": "group"', '"chat_type": "direct"',
    ]
    if any(marker in system_text for marker in human_channel_markers):
        return True

    # OpenClaw-Session ohne explizite Channel-Marker -> für Nutzer standardmäßig als human behandeln
    if "you are a personal assistant running inside openclaw." in system_text:
        return True

    # Sonst eher API-/Maschinenverkehr
    return False

# --- Config ---
from academicai.config import (
    INSECURE_PROXY_KEYS,
    _INSECURE_PROXY_KEYS,
    _validate_proxy_api_key,
    validate_config,
    get_settings,
    Settings,
    PORT,
    API_KEY,
    BASE_URL,
    HEALTH_CHECK_BACKEND,
    HEALTH_CHECK_TIMEOUT_SECONDS,
    ENABLE_COST_MONITORING,
    COST_CACHE_FILE,
    COST_CACHE_TTL_SECONDS,
    COST_REFRESH_TIMEOUT_SECONDS,
    MAX_MESSAGES,
    MAX_TOOLS,
    MAX_MESSAGE_TEXT_CHARS,
    MAX_TOOL_SCHEMA_CHARS,
    MAX_REQUEST_JSON_CHARS,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_WINDOW_SECONDS,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_TOOL_TEMPERATURE,
    DEFAULT_CHAT_VERBOSITY,
    DEFAULT_TOOL_VERBOSITY,
    DEFAULT_TOOL_REASONING_EFFORT,
    ENABLE_HUMANIZATION_PASS,
    HUMANIZATION_MODEL,
    HUMANIZATION_TEMPERATURE,
    STREAM_CHUNK_DELAY_MS,
    DEBUG_DUMPS,
    ALLOWED_MODELS,
    PID_FILE,
    LOG_FILE_PATH,
    ERR_FILE_PATH,
    RETRY_MAX,
    RETRY_BASE_MS,
)

_cost_lock = threading.Lock()
_cost_refresh_in_flight = False


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_ts(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        value = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except Exception:
        return None


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _extract_cost_summary(payload: dict) -> dict:
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


def _read_cost_cache() -> dict:
    p = Path(COST_CACHE_FILE)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_cost_cache(cache: dict) -> None:
    p = Path(COST_CACHE_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_cost_cache_stale(cache: dict) -> bool:
    ts = _parse_iso_ts(str(cache.get("updated_at", "")))
    if ts is None:
        return True
    return (datetime.now(timezone.utc) - ts).total_seconds() > COST_CACHE_TTL_SECONDS


def _build_cost_headers(cache: dict) -> dict:
    if not ENABLE_COST_MONITORING:
        return {}
    if not cache:
        return {}
    headers = {
        "X-AcademicAI-Cost-Stale": "true" if _is_cost_cache_stale(cache) else "false",
    }
    updated_at = str(cache.get("updated_at", "")).strip()
    if updated_at:
        headers["X-AcademicAI-Cost-Updated-At"] = updated_at

    total_cost = _safe_float(cache.get("total_cost"))
    if total_cost is not None:
        headers["X-AcademicAI-Total-Cost"] = f"{total_cost:.6f}".rstrip("0").rstrip(".")

    total_clients = cache.get("total_clients")
    if isinstance(total_clients, int):
        headers["X-AcademicAI-Total-Clients"] = str(total_clients)

    cost_entries = cache.get("cost_entries")
    if isinstance(cost_entries, int):
        headers["X-AcademicAI-Cost-Entries"] = str(cost_entries)

    return headers


def _fetch_cost_snapshot() -> dict:
    if not ENABLE_COST_MONITORING:
        return {}

    base_url = get_base_url().rstrip("/")
    cost_url = f"{base_url}/api/v1/cost/"
    headers = dict(get_headers() or {})
    headers.setdefault("Accept", "application/json")

    with httpx.Client(timeout=COST_REFRESH_TIMEOUT_SECONDS, follow_redirects=True) as client:
        resp = client.get(cost_url, headers=headers)
        resp.raise_for_status()
        payload = resp.json()

    summary = _extract_cost_summary(payload)
    return {
        "updated_at": _now_utc_iso(),
        "source": "live",
        "raw": payload,
        **summary,
    }


def _refresh_cost_cache_sync() -> dict:
    if not ENABLE_COST_MONITORING:
        return _read_cost_cache()

    with _cost_lock:
        fresh = _fetch_cost_snapshot()
        if fresh:
            _write_cost_cache(fresh)
        return fresh


async def _refresh_cost_cache_background() -> None:
    global _cost_refresh_in_flight
    try:
        await run_in_threadpool(_refresh_cost_cache_sync)
    except Exception as e:
        log.warning(f"cost refresh failed: {e}")
    finally:
        _cost_refresh_in_flight = False


def _get_cost_cache_with_lazy_refresh() -> dict:
    global _cost_refresh_in_flight
    cache = _read_cost_cache()

    if not ENABLE_COST_MONITORING:
        return cache

    if _is_cost_cache_stale(cache) and not _cost_refresh_in_flight:
        try:
            loop = asyncio.get_running_loop()
            _cost_refresh_in_flight = True
            loop.create_task(_refresh_cost_cache_background())
        except RuntimeError:
            # Kein laufender Loop (z.B. in unit tests) -> synchron vermeiden
            pass

    return cache


def _build_humanization_messages(original_user_query: str, structured_content: str) -> list:
    """Prompt für den optionalen zweiten LLM-Pass (Humanisierung)."""
    system_msg = {
        "role": "system",
        "content": (
            "You rewrite structured tool output into a natural final answer for a human chat. "
            "Return only the final answer text for the user. "
            "Do NOT include JSON, code blocks, field names, metadata, or debug info."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Original user question:\n{original_user_query.strip() or '-'}\n\n"
            f"Structured/tool-derived result:\n{structured_content.strip()}\n\n"
            "Task: Write a concise, natural-language final reply for the user."
        ),
    }
    return [system_msg, user_msg]


async def _run_humanization_pass(model: str, original_user_query: str, structured_content: str) -> Optional[str]:
    """Führt optionalen zweiten LLM-Pass aus und liefert finalen Text zurück."""
    try:
        human_model = HUMANIZATION_MODEL or model
        resp = await run_in_threadpool(
            academicai.completion,
            model=human_model,
            messages=_build_humanization_messages(original_user_query, structured_content),
            temperature=HUMANIZATION_TEMPERATURE,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as e:
        log.warning(f"humanization pass failed, fallback to first-pass content: {e}")
        return None


def _write_pid_file() -> None:
    try:
        PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    except Exception as e:
        log.warning(f"could not write pid file {PID_FILE}: {e}")


def _cleanup_pid_file() -> None:
    try:
        if not PID_FILE.exists():
            return
        raw = PID_FILE.read_text(encoding="utf-8").strip()
        if raw and raw != str(os.getpid()):
            return
        PID_FILE.unlink(missing_ok=True)
    except Exception as e:
        log.warning(f"could not cleanup pid file {PID_FILE}: {e}")


def _check_backend_health() -> dict:
    if not HEALTH_CHECK_BACKEND:
        return {"enabled": False, "ok": None}

    started = time.perf_counter()
    try:
        endpoint = f"{get_base_url().rstrip('/')}/api/v1/llm/models"
        headers = dict(get_headers() or {})
        with httpx.Client(timeout=HEALTH_CHECK_TIMEOUT_SECONDS) as client:
            resp = client.get(endpoint, headers=headers)
        latency_ms = int((time.perf_counter() - started) * 1000)
        ok = resp.status_code == 200
        out = {
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

# --- Setup ---
from logging.handlers import TimedRotatingFileHandler

log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

# TimedRotatingFileHandler for general logs (INFO and above), rotated daily, 30 days retention
log_file_path = LOG_FILE_PATH
info_handler = TimedRotatingFileHandler(log_file_path, when="D", interval=1, backupCount=30, encoding="utf-8")
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(log_formatter)

# TimedRotatingFileHandler for error logs (ERROR and above), rotated daily, 30 days retention
err_file_path = ERR_FILE_PATH
error_handler = TimedRotatingFileHandler(err_file_path, when="D", interval=1, backupCount=30, encoding="utf-8")
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_formatter)

# Console logger for debug runs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

# Root logger setup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(info_handler)
root_logger.addHandler(error_handler)
root_logger.addHandler(console_handler)

# Configure uvicorn loggers to use rotating handlers
for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    ulog = logging.getLogger(uvicorn_logger_name)
    ulog.handlers = []
    ulog.addHandler(info_handler)
    ulog.addHandler(error_handler)
    ulog.addHandler(console_handler)
    ulog.propagate = False

log = logging.getLogger("academicai-proxy")

app = FastAPI(
    title="AcademicAI Proxy",
    description="OpenAI-kompatibler Proxy für AcademicAI",
    version="1.0.0",
)

security = HTTPBearer(auto_error=False)


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials


# --- Health ---


@app.on_event("startup")
def _on_startup() -> None:
    validate_config()
    _write_pid_file()


@app.on_event("shutdown")
def _on_shutdown() -> None:
    _cleanup_pid_file()

@app.get("/health")
def health():
    backend = _check_backend_health()
    status = "ok"
    if backend.get("enabled") and backend.get("ok") is False:
        status = "degraded"
    return {"status": status, "service": "academicai-proxy", "backend": backend}


@app.get("/internal/cost-status")
def cost_status(key: str = Depends(verify_key)):
    cache = _get_cost_cache_with_lazy_refresh()
    return {
        "enabled": ENABLE_COST_MONITORING,
        "total_cost": _safe_float(cache.get("total_cost")),
        "total_clients": cache.get("total_clients"),
        "cost_entries": cache.get("cost_entries"),
        "updated_at": cache.get("updated_at"),
        "is_stale": _is_cost_cache_stale(cache) if cache else True,
        "source": cache.get("source", "cache" if cache else "none"),
    }


# --- Models ---

@app.get("/v1/models")
async def list_models(key: str = Depends(verify_key)):
    try:
        return await run_in_threadpool(academicai.get_models)
    except Exception as e:
        log.error(f"get_models failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# --- Chat Completions ---

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, key: str = Depends(verify_key)):
    try:
        body = await request.json()
    except Exception:
        log.warning("Chat request rejected (400): invalid JSON body")
        raise HTTPException(status_code=400, detail="invalid JSON body")

    validate_request_json_size(body, max_chars=MAX_REQUEST_JSON_CHARS)
    _validate_chat_request_body(body)
    _enforce_chat_rate_limit(request, key)

    # Vollständiges Request-Dump für Debugging (optional via Env)
    if DEBUG_DUMPS:
        import json as _json
        _dump_path = os.path.join(os.path.dirname(__file__), "last_request.json")
        try:
            with open(_dump_path, "w", encoding="utf-8") as _f:
                _json.dump(redact_sensitive(body), _f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    model = body.get("model")
    messages = list(body.get("messages") or [])

    # Top-level "system" Parameter (z.B. von Anthropic-style Clients) → in messages einfügen
    top_level_system = body.get("system")
    if top_level_system and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": top_level_system})

    original_user_query = _last_user_text(messages)

    # Tools extrahieren — werden via Prompt-Injection emuliert
    tools = body.get("tools") or body.get("functions") or []
    has_tools = bool(tools)
    if ("tool_choice" in body) and not has_tools:
        log.warning("tool_choice provided without tools; ignoring tool emulation for this request")
    log.info(f"incoming: model={model} stream={body.get('stream')} roles={[m.get('role') for m in messages]} tools={len(tools)} has_tools={has_tools}")

    # Bei Follow-up nach Tool-Result finale Antwort stärker priorisieren
    messages = _apply_post_tool_guard(messages, has_tools=has_tools)

    # Tool-Definitionen in System-Prompt injizieren
    if tools:
        messages = inject_tools_into_messages(messages, tools)

    want_stream = bool(body.get("stream"))
    human_target_hint = _is_human_readable_target(messages)
    cost_cache = _get_cost_cache_with_lazy_refresh()
    response_headers = _build_cost_headers(cost_cache)

    # Optionale Parameter weiterreichen — nur bekannte, AcademicAI-sichere Felder
    # tools / tool_choice / functions werden via Prompt-Injection emuliert (nicht nativ weitergegeben)
    optional = {}
    for field in [
        "temperature", "max_tokens", "max_completion_tokens",
        "frequency_penalty", "presence_penalty",
        "reasoning_effort", "verbosity", "seed", "stop",
    ]:
        if field in body:
            optional[field] = body[field]

    # Sinnvolle Proxy-Defaults (nur falls Client nichts gesetzt hat)
    if has_tools:
        optional.setdefault("temperature", DEFAULT_TOOL_TEMPERATURE)
        # GPT-5-Modelle profitieren bei Emulation von knapper, deterministischerem Stil
        if model and "gpt-5" in model:
            optional.setdefault("verbosity", DEFAULT_TOOL_VERBOSITY)
            optional.setdefault("reasoning_effort", DEFAULT_TOOL_REASONING_EFFORT)
    elif human_target_hint:
        optional.setdefault("temperature", DEFAULT_CHAT_TEMPERATURE)
        if model and "gpt-5" in model:
            optional.setdefault("verbosity", DEFAULT_CHAT_VERBOSITY)

    # response_format: bei Tool-Emulation JSON-Mode erzwingen,
    # sonst Wert aus Request durchreichen (ausser json_schema)
    if tools:
        optional["response_format"] = {"type": "json_object"}
    elif "response_format" in body:
        rf = body.get("response_format")
        if isinstance(rf, dict):
            if rf.get("type") != "json_schema":
                optional["response_format"] = rf
        else:
            log.warning("ignoring non-dict response_format from client")

    # tailoredAiId via extra_body
    if "extra_body" in body and "tailoredAiId" in body["extra_body"]:
        optional["extra_body"] = {"tailoredAiId": body["extra_body"]["tailoredAiId"]}

    try:
        response = await run_in_threadpool(academicai.completion, model=model, messages=messages, **optional)
    except Exception as e:
        log.error(f"completion failed: model={model} error={e}")
        raise HTTPException(status_code=502, detail=str(e))

    completion_id = response.id
    created_ts = response.created
    resp_model = response.model
    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = choice.finish_reason or "stop"
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    # JSON-Mode Response verarbeiten (nur wenn Tools im Request waren)
    tool_calls_data = []
    human_target = human_target_hint
    if has_tools:
        tool_calls_data = parse_tool_calls(content)
        if tool_calls_data:
            names = [c.get("name", "?") for c in tool_calls_data]
            log.info(f"tool_call(s) detected: count={len(tool_calls_data)} names={names}")
        else:
            # Kein Tool-Call — entweder {"action":"respond",...} oder Fallback
            extracted = extract_respond_content(content)
            if extracted is not None:
                log.info(f"json_mode respond: content_len={len(extracted)}")
                content = extracted

                # GPT-5 liefert teils action=respond mit JSON-String in content.
                # Auf Human-Targets trotzdem in natürlich lesbaren Text umformen.
                if human_target:
                    humanized_from_content = format_arbitrary_json_for_humans(content)
                    if humanized_from_content is not None:
                        log.warning("json_mode respond: JSON-string content -> human text (human target)")
                        content = humanized_from_content
            else:
                # Letzter Fallback nur für human-readable Targets.
                # Für maschinelle Flows (z.B. cron) bleibt raw content erhalten.
                if human_target:
                    human_text = format_arbitrary_json_for_humans(content)
                    if human_text is not None:
                        log.warning(f"json_mode: arbitrary JSON -> human text (human target): {content[:80]}")
                        content = human_text
                    else:
                        # Fallback-Fallback: falls Rendern scheitert, wenigstens lesbar
                        codeblock = format_arbitrary_json_as_codeblock(content)
                        if codeblock is not None:
                            log.warning(f"json_mode: arbitrary JSON -> code block fallback (human target): {content[:80]}")
                            content = codeblock
                        else:
                            log.warning(f"json_mode parse failed, using raw content: {content[:120]}")
                else:
                    log.warning("json_mode: arbitrary JSON on non-human target, keeping raw content")

    # Optionaler zweiter Pass: natürliche Endantwort für Human-Channels
    if (
        ENABLE_HUMANIZATION_PASS
        and human_target
        and has_tools
        and not tool_calls_data
        and (content or "").strip()
    ):
        humanized = await _run_humanization_pass(model=resp_model, original_user_query=original_user_query, structured_content=content)
        if humanized:
            log.info(f"humanization pass applied: len_before={len(content)} len_after={len(humanized)}")
            content = humanized

    # Wenn Streaming gewünscht: Antwort als SSE emulieren
    if want_stream:
        def sse_generator():
            if tool_calls_data:
                # Tool-Call-Chunks im OpenAI-Streaming-Format
                for chunk in build_tool_calls_sse_chunks(completion_id, created_ts, resp_model, tool_calls_data):
                    if STREAM_CHUNK_DELAY_MS > 0:
                        time.sleep(STREAM_CHUNK_DELAY_MS / 1000.0)
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Normaler Text-Response als SSE
                # Chunk 1: role delta
                if STREAM_CHUNK_DELAY_MS > 0:
                    time.sleep(STREAM_CHUNK_DELAY_MS / 1000.0)
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': resp_model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
                # Chunk 2: Content
                if STREAM_CHUNK_DELAY_MS > 0:
                    time.sleep(STREAM_CHUNK_DELAY_MS / 1000.0)
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': resp_model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
                # Chunk 3: finish
                if STREAM_CHUNK_DELAY_MS > 0:
                    time.sleep(STREAM_CHUNK_DELAY_MS / 1000.0)
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': resp_model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream", headers=response_headers)

    # Kein Streaming: normaler JSON-Response
    if tool_calls_data:
        return JSONResponse(
            content=build_tool_calls_response(completion_id, created_ts, resp_model, tool_calls_data, usage),
            headers=response_headers,
        )

    return JSONResponse(
        content={
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": resp_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        },
        headers=response_headers,
    )


# --- Start ---

if __name__ == "__main__":
    validate_config()
    startup_banner = r"""
    _                _                 _      _      ___ 
   /_\  __ __ _  __| |___ _ __  _  _ _| |_ __(_)  _|_  |
  / _ \/ _/ _` |/ _` / -_) '  \| |/ _` |  _/ _| | | |_  |
 /_/ \_\__\__,_|\__,_\___|_|_|_|_|\__,_|\__\__|_| | /___|
 --------------------------------------------------------
             AcademicAI Proxy Adapter - v1.0.0
 --------------------------------------------------------
"""
    sys.stdout.write(startup_banner)
    sys.stdout.write(f"  [Host/Port]     127.0.0.1:{PORT}\n")
    sys.stdout.write(f"  [API Key]       {'Active' if API_KEY else 'Missing/Insecure'}\n")
    sys.stdout.write(f"  [Log Rotation]  Active (30-day retention)\n")
    sys.stdout.write(f"  [Humanization]  {'Active (temp: ' + str(HUMANIZATION_TEMPERATURE) + ')' if ENABLE_HUMANIZATION_PASS else 'Disabled'}\n")
    sys.stdout.write(f"  [Request Limits]  {MAX_MESSAGES} msg / {MAX_TOOLS} tools / {MAX_MESSAGE_TEXT_CHARS} msg-chars / {MAX_REQUEST_JSON_CHARS} json-chars\n")
    sys.stdout.write(f"  [Test Port]     11436 (isolated)\n")
    sys.stdout.write(" --------------------------------------------------------\n\n")
    sys.stdout.flush()

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
