"""
AcademicAI FastAPI Application Factory and Modern Lifespan Management.

Provides:
- Modern `@asynccontextmanager async def lifespan(app: FastAPI)` replacing deprecated `@app.on_event`
- Application factory `create_app() -> FastAPI`
- Module-level `app = create_app()`
- Route handlers: /health, /internal/cost-status, /v1/models, /v1/chat/completions
- Dynamic configuration resolution via `_get_setting` for monkeypatch backward compatibility
"""

import inspect
import json
import logging
import os
import sys
import time
from collections import namedtuple
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

from academicai.config import (
    API_KEY,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_CHAT_VERBOSITY,
    DEFAULT_TOOL_REASONING_EFFORT,
    DEFAULT_TOOL_TEMPERATURE,
    DEFAULT_TOOL_VERBOSITY,
    ENABLE_HUMANIZATION_PASS,
    ENABLE_LOCAL_COST_TRACKING,
    MODEL_PRICING_CACHE_TTL_SECONDS,
    LOCAL_COST_CACHE_FILE,
    LOCAL_COST_HISTORY_LIMIT,
    MAX_REQUEST_JSON_CHARS,
    STREAM_CHUNK_DELAY_MS,
    validate_config,
)
from academicai.cost_calculation import (
    calculate_request_cost,
    get_model_catalog,
    get_pricing_cache,
    ModelCatalog,
    RequestCost,
)
from academicai.local_cost_tracker import (
    get_local_cost_store,
    LocalCostStore,
)
from academicai.cost_monitoring import (
    build_cost_headers,
    get_cost_cache_with_lazy_refresh,
    get_cost_status_payload,
)
from academicai.errors import AcademicAIError
from academicai.humanization import (
    is_human_readable_target,
    last_user_text,
    run_humanization_pass,
)
from academicai.logging_config import log
from academicai.request_guards import (
    enforce_chat_rate_limit,
    validate_chat_request_body,
    validate_request_json_size,
    validate_responses_request_body,
)
from academicai.responses import (
    build_responses_output,
    build_responses_sse_events,
    normalize_responses_request,
)
from academicai.runtime import (
    check_backend_health,
    cleanup_pid_file,
    get_health_payload,
    write_pid_file,
)
from academicai.security import redact_sensitive
from academicai.tool_emulation import (
    apply_post_tool_guard,
    build_tool_calls_response,
    build_tool_calls_sse_chunks,
    extract_respond_content,
    format_arbitrary_json_as_codeblock,
    format_arbitrary_json_for_humans,
    inject_tools_into_messages,
    parse_tool_calls,
)

# Underscore aliases for backward compatibility and test monkeypatching
_write_pid_file = write_pid_file
_cleanup_pid_file = cleanup_pid_file
_check_backend_health = check_backend_health
_get_cost_cache_with_lazy_refresh = get_cost_cache_with_lazy_refresh
_build_cost_headers = build_cost_headers
_validate_chat_request_body = validate_chat_request_body
_validate_responses_request_body = validate_responses_request_body
_enforce_chat_rate_limit = enforce_chat_rate_limit
_last_user_text = last_user_text
_apply_post_tool_guard = apply_post_tool_guard
_is_human_readable_target = is_human_readable_target
_run_humanization_pass = run_humanization_pass
_calculate_request_cost = calculate_request_cost
_get_pricing_cache = get_pricing_cache
_get_local_cost_store = get_local_cost_store

_INITIAL_DEFAULTS: dict[str, Any] = {
    "validate_config": validate_config,
    "write_pid_file": write_pid_file,
    "_write_pid_file": write_pid_file,
    "cleanup_pid_file": cleanup_pid_file,
    "_cleanup_pid_file": cleanup_pid_file,
    "check_backend_health": check_backend_health,
    "_check_backend_health": check_backend_health,
    "get_health_payload": get_health_payload,
    "get_cost_cache_with_lazy_refresh": get_cost_cache_with_lazy_refresh,
    "_get_cost_cache_with_lazy_refresh": get_cost_cache_with_lazy_refresh,
    "get_cost_status_payload": get_cost_status_payload,
    "build_cost_headers": build_cost_headers,
    "_build_cost_headers": build_cost_headers,
    "calculate_request_cost": calculate_request_cost,
    "_calculate_request_cost": calculate_request_cost,
    "get_pricing_cache": get_pricing_cache,
    "_get_pricing_cache": get_pricing_cache,
    "get_local_cost_store": get_local_cost_store,
    "_get_local_cost_store": get_local_cost_store,
    "validate_request_json_size": validate_request_json_size,
    "validate_chat_request_body": validate_chat_request_body,
    "_validate_chat_request_body": validate_chat_request_body,
    "validate_responses_request_body": validate_responses_request_body,
    "_validate_responses_request_body": validate_responses_request_body,
    "normalize_responses_request": normalize_responses_request,
    "build_responses_output": build_responses_output,
    "build_responses_sse_events": build_responses_sse_events,
    "enforce_chat_rate_limit": enforce_chat_rate_limit,
    "_enforce_chat_rate_limit": enforce_chat_rate_limit,
    "last_user_text": last_user_text,
    "_last_user_text": last_user_text,
    "apply_post_tool_guard": apply_post_tool_guard,
    "_apply_post_tool_guard": apply_post_tool_guard,
    "is_human_readable_target": is_human_readable_target,
    "_is_human_readable_target": is_human_readable_target,
    "run_humanization_pass": run_humanization_pass,
    "_run_humanization_pass": run_humanization_pass,
    "API_KEY": API_KEY,
    "MAX_REQUEST_JSON_CHARS": MAX_REQUEST_JSON_CHARS,
    "ENABLE_HUMANIZATION_PASS": ENABLE_HUMANIZATION_PASS,
    "ENABLE_LOCAL_COST_TRACKING": ENABLE_LOCAL_COST_TRACKING,
    "DEFAULT_CHAT_TEMPERATURE": DEFAULT_CHAT_TEMPERATURE,
    "DEFAULT_TOOL_TEMPERATURE": DEFAULT_TOOL_TEMPERATURE,
    "DEFAULT_CHAT_VERBOSITY": DEFAULT_CHAT_VERBOSITY,
    "DEFAULT_TOOL_VERBOSITY": DEFAULT_TOOL_VERBOSITY,
    "DEFAULT_TOOL_REASONING_EFFORT": DEFAULT_TOOL_REASONING_EFFORT,
    "STREAM_CHUNK_DELAY_MS": STREAM_CHUNK_DELAY_MS,
    "DEBUG_DUMPS": False,
}

security = HTTPBearer(auto_error=False)


def _get_setting(name: str, default: Any = None) -> Any:
    """
    Liefert eine Konfigurationseinstellung oder ein Callable.
    Priorität:
    1. Dynamisch gesetztes Attribut auf server (falls monkeypatched / ungleich _INITIAL_DEFAULTS)
    2. Dynamisch gesetztes Attribut auf academicai.app (falls monkeypatched / ungleich _INITIAL_DEFAULTS)
    3. Attribut auf server (falls vorhanden)
    4. Attribut auf academicai.app (falls vorhanden)
    5. Modul academicai.config (falls vorhanden)
    6. Übergebener default-Wert
    """
    server_mod = sys.modules.get("server")
    app_mod = sys.modules.get("academicai.app")

    s_val = getattr(server_mod, name, None) if server_mod is not None else None
    a_val = getattr(app_mod, name, None) if app_mod is not None else None

    init = _INITIAL_DEFAULTS.get(name, default)

    # 1. Monkeypatch auf server erkannt?
    if s_val is not None and s_val != init:
        return s_val

    # 2. Monkeypatch auf academicai.app erkannt?
    if a_val is not None and a_val != init:
        return a_val

    # 3. Vorhanden auf server
    if s_val is not None:
        return s_val

    # 4. Vorhanden auf academicai.app
    if a_val is not None:
        return a_val

    # 5. Modul academicai.config
    import academicai.config as config
    if hasattr(config, name):
        return getattr(config, name)

    return default


def _resolve_completion():
    server_mod = sys.modules.get("server")
    if server_mod is not None:
        if hasattr(server_mod, "academicai") and hasattr(server_mod.academicai, "completion"):
            return server_mod.academicai.completion
        if hasattr(server_mod, "completion"):
            return server_mod.completion
    import academicai
    return getattr(academicai, "completion")


def _resolve_get_models():
    server_mod = sys.modules.get("server")
    if server_mod is not None:
        if hasattr(server_mod, "academicai") and hasattr(server_mod.academicai, "get_models"):
            return server_mod.academicai.get_models
        if hasattr(server_mod, "get_models"):
            return server_mod.get_models
    import academicai
    return getattr(academicai, "get_models")


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    api_key = _get_setting("API_KEY", API_KEY)
    if credentials is None or credentials.credentials != api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials


# --- Lifespan Handler ---


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Startup: Config validieren & PID-Datei schreiben
    val_cfg = _get_setting("validate_config", validate_config)
    if callable(val_cfg):
        val_cfg()

    write_pid = None
    s_pid = _get_setting("_write_pid_file")
    if s_pid != _INITIAL_DEFAULTS.get("_write_pid_file"):
        write_pid = s_pid
    else:
        write_pid = _get_setting("write_pid_file", write_pid_file)
    if callable(write_pid):
        write_pid()

    try:
        yield
    finally:
        # Shutdown: PID-Datei bereinigen
        cleanup_pid = None
        s_clean = _get_setting("_cleanup_pid_file")
        if s_clean != _INITIAL_DEFAULTS.get("_cleanup_pid_file"):
            cleanup_pid = s_clean
        else:
            cleanup_pid = _get_setting("cleanup_pid_file", cleanup_pid_file)
        if callable(cleanup_pid):
            cleanup_pid()


# --- Route Handlers ---


def health():
    checker = None
    s_val = _get_setting("_check_backend_health")
    if s_val != _INITIAL_DEFAULTS.get("_check_backend_health"):
        checker = s_val
    else:
        checker = _get_setting("check_backend_health", check_backend_health)
    payload_fn = _get_setting("get_health_payload", get_health_payload)
    backend = checker()
    return payload_fn(backend)


def cost_status(key: str = Depends(verify_key)):
    cache_fn = None
    s_val = _get_setting("_get_cost_cache_with_lazy_refresh")
    if s_val != _INITIAL_DEFAULTS.get("_get_cost_cache_with_lazy_refresh"):
        cache_fn = s_val
    else:
        cache_fn = _get_setting("get_cost_cache_with_lazy_refresh", get_cost_cache_with_lazy_refresh)
    payload_fn = _get_setting("get_cost_status_payload", get_cost_status_payload)
    cache = cache_fn()
    base_payload = payload_fn(cache)

    # Local cost tracking status
    store_fn = _get_setting("_get_local_cost_store") or _get_setting(
        "get_local_cost_store", get_local_cost_store
    )
    store = store_fn()
    local_status = store.get_status_payload()

    # Model catalog & pricing cache status
    catalog_fn = (
        _get_setting("_get_model_catalog")
        or _get_setting("get_model_catalog")
        or _get_setting("_get_pricing_cache")
        or _get_setting("get_pricing_cache", get_model_catalog)
    )
    catalog = catalog_fn()
    catalog_status = catalog.get_status()
    local_status["model_catalog"] = catalog_status
    local_status["pricing_cache"] = catalog_status

    # Merged payload: preserves all root keys for backward compatibility
    # and adds structured sections for both backend monitoring and local tracking.
    payload = dict(base_payload)
    payload["backend_cost_monitoring"] = dict(base_payload)
    payload["local_cost_tracking"] = local_status
    return payload


async def list_models(key: str = Depends(verify_key)):
    try:
        models_fn = _resolve_get_models()
        if inspect.iscoroutinefunction(models_fn):
            return await models_fn()
        return await run_in_threadpool(models_fn)
    except (HTTPException, AcademicAIError):
        raise
    except Exception as e:
        log.error(f"get_models failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))


CanonicalResult = namedtuple(
    "CanonicalResult",
    [
        "completion_id",
        "created_ts",
        "model",
        "content",
        "finish_reason",
        "usage",
        "tool_calls_data",
    ],
)


async def _execute_completion_pipeline(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: Any = None,
    optional_params: Optional[dict[str, Any]] = None,
    stream: bool = False,
) -> CanonicalResult:
    """
    Kanonische Completion-Pipeline für Chat-Completions und Responses API:
    - User-Query-Erkennung
    - Post-Tool-Guard (Fokus auf finale Antwort nach Tool-Ergebnissen)
    - Tool-Emulation (TypeScript-Schema Prompt-Injection & tool_choice)
    - Fallback-Defaults für Temperature, Verbosity und Reasoning Effort
    - Backend-Aufruf via _resolve_completion
    - Tool-Call Parsing, Respond-Content-Extraktion und Fallbacks
    - Optionaler Humanisierungs-Pass für Human-Targets
    """
    user_text_fn = _get_setting("_last_user_text") or _get_setting("last_user_text", last_user_text)
    original_user_query = user_text_fn(messages)

    has_tools = bool(tools)
    if tool_choice is not None and not has_tools:
        log.warning("tool_choice provided without tools; ignoring tool emulation for this request")
    log.info(
        f"incoming completion: model={model} stream={stream} "
        f"roles={[m.get('role') for m in messages]} tools={len(tools)} has_tools={has_tools}"
    )

    # Bei Follow-up nach Tool-Result finale Antwort stärker priorisieren
    post_tool_guard_fn = _get_setting("_apply_post_tool_guard") or _get_setting(
        "apply_post_tool_guard", apply_post_tool_guard
    )
    messages = post_tool_guard_fn(messages, has_tools=has_tools)

    # Tool-Definitionen in System-Prompt injizieren
    if tools:
        inject_fn = _get_setting("inject_tools_into_messages", inject_tools_into_messages)
        messages = inject_fn(messages, tools, tool_choice=tool_choice)

    human_target_fn = _get_setting("_is_human_readable_target") or _get_setting(
        "is_human_readable_target", is_human_readable_target
    )
    human_target_hint = human_target_fn(messages)

    optional = dict(optional_params or {})

    # Sinnvolle Proxy-Defaults (nur falls Client nichts gesetzt hat)
    if has_tools:
        optional.setdefault("temperature", _get_setting("DEFAULT_TOOL_TEMPERATURE", DEFAULT_TOOL_TEMPERATURE))
        if model and "gpt-5" in model:
            optional.setdefault("verbosity", _get_setting("DEFAULT_TOOL_VERBOSITY", DEFAULT_TOOL_VERBOSITY))
            optional.setdefault("reasoning_effort", _get_setting("DEFAULT_TOOL_REASONING_EFFORT", DEFAULT_TOOL_REASONING_EFFORT))
    elif human_target_hint:
        optional.setdefault("temperature", _get_setting("DEFAULT_CHAT_TEMPERATURE", DEFAULT_CHAT_TEMPERATURE))
        if model and "gpt-5" in model:
            optional.setdefault("verbosity", _get_setting("DEFAULT_CHAT_VERBOSITY", DEFAULT_CHAT_VERBOSITY))

    # response_format: bei Tool-Emulation JSON-Mode erzwingen
    if tools:
        optional["response_format"] = {"type": "json_object"}
    elif "response_format" in optional:
        rf = optional.get("response_format")
        if isinstance(rf, dict):
            if rf.get("type") == "json_schema":
                optional.pop("response_format", None)
        else:
            log.warning("ignoring non-dict response_format from client")
            optional.pop("response_format", None)

    comp_fn = _resolve_completion()
    if inspect.iscoroutinefunction(comp_fn):
        response = await comp_fn(model=model, messages=messages, **optional)
    else:
        response = await run_in_threadpool(comp_fn, model=model, messages=messages, **optional)

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
            extracted = extract_respond_content(content)
            if extracted is not None:
                log.info(f"json_mode respond: content_len={len(extracted)}")
                content = extracted
                if human_target:
                    humanized_from_content = format_arbitrary_json_for_humans(content)
                    if humanized_from_content is not None:
                        log.warning("json_mode respond: JSON-string content -> human text (human target)")
                        content = humanized_from_content
            else:
                if human_target:
                    human_text = format_arbitrary_json_for_humans(content)
                    if human_text is not None:
                        log.warning(f"json_mode: arbitrary JSON -> human text (human target): {content[:80]}")
                        content = human_text
                    else:
                        codeblock = format_arbitrary_json_as_codeblock(content)
                        if codeblock is not None:
                            log.warning(f"json_mode: arbitrary JSON -> code block fallback (human target): {content[:80]}")
                            content = codeblock
                        else:
                            log.warning(f"json_mode parse failed, using raw content: {content[:120]}")
                else:
                    log.warning("json_mode: arbitrary JSON on non-human target, keeping raw content")

    # Optionaler zweiter Pass: natürliche Endantwort für Human-Channels
    enable_humanization = _get_setting("ENABLE_HUMANIZATION_PASS", ENABLE_HUMANIZATION_PASS)
    if (
        enable_humanization
        and human_target
        and has_tools
        and not tool_calls_data
        and (content or "").strip()
    ):
        humanize_fn = _get_setting("_run_humanization_pass") or _get_setting("run_humanization_pass", run_humanization_pass)
        humanized = await humanize_fn(model=resp_model, original_user_query=original_user_query, structured_content=content)
        if humanized:
            log.info(f"humanization pass applied: len_before={len(content)} len_after={len(humanized)}")
            content = humanized

    return CanonicalResult(
        completion_id=completion_id,
        created_ts=created_ts,
        model=resp_model,
        content=content,
        finish_reason=finish_reason,
        usage=usage,
        tool_calls_data=tool_calls_data,
    )


async def chat_completions(request: Request, key: str = Depends(verify_key)):
    try:
        body = await request.json()
    except Exception:
        log.warning("Chat request rejected (400): invalid JSON body")
        raise HTTPException(status_code=400, detail="invalid JSON body")

    max_req_chars = _get_setting("MAX_REQUEST_JSON_CHARS", MAX_REQUEST_JSON_CHARS)
    size_validator = _get_setting("validate_request_json_size", validate_request_json_size)
    size_validator(body, max_chars=max_req_chars)

    body_validator = _get_setting("_validate_chat_request_body") or _get_setting("validate_chat_request_body", validate_chat_request_body)
    body_validator(body)

    rate_limiter = _get_setting("_enforce_chat_rate_limit") or _get_setting("enforce_chat_rate_limit", enforce_chat_rate_limit)
    rate_limiter(request, key)

    # Vollständiges Request-Dump für Debugging (optional via Env)
    debug_dumps = _get_setting("DEBUG_DUMPS", False)
    if debug_dumps:
        _dump_path = Path(__file__).resolve().parent.parent / "last_request.json"
        try:
            with open(_dump_path, "w", encoding="utf-8") as _f:
                json.dump(redact_sensitive(body), _f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    model = body.get("model")
    messages = list(body.get("messages") or [])

    # Top-level "system" Parameter (z.B. von Anthropic-style Clients) → in messages einfügen
    top_level_system = body.get("system")
    if top_level_system and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": top_level_system})

    tools = body.get("tools") or body.get("functions") or []
    tool_choice = body.get("tool_choice")
    want_stream = bool(body.get("stream"))

    cost_cache_fn = _get_setting("_get_cost_cache_with_lazy_refresh") or _get_setting("get_cost_cache_with_lazy_refresh", get_cost_cache_with_lazy_refresh)
    cost_cache = cost_cache_fn()
    cost_headers_fn = _get_setting("_build_cost_headers") or _get_setting("build_cost_headers", build_cost_headers)
    response_headers = cost_headers_fn(cost_cache)

    # Optionale Parameter weiterreichen — nur bekannte, AcademicAI-sichere Felder
    optional = {}
    for field in [
        "temperature", "max_tokens", "max_completion_tokens",
        "frequency_penalty", "presence_penalty",
        "reasoning_effort", "verbosity", "seed", "stop",
    ]:
        if field in body:
            optional[field] = body[field]

    if "response_format" in body:
        optional["response_format"] = body["response_format"]

    if "extra_body" in body and "tailoredAiId" in body["extra_body"]:
        optional["extra_body"] = {"tailoredAiId": body["extra_body"]["tailoredAiId"]}

    try:
        result = await _execute_completion_pipeline(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            optional_params=optional,
            stream=want_stream,
        )
    except (HTTPException, AcademicAIError):
        raise
    except Exception as e:
        log.error(f"completion failed: model={model} error={e}")
        raise HTTPException(status_code=502, detail=str(e))

    # Lokale Kostenberechnung & Header
    enable_local_cost = bool(_get_setting("ENABLE_LOCAL_COST_TRACKING", ENABLE_LOCAL_COST_TRACKING))
    if enable_local_cost:
        cost_calc_fn = _get_setting("_calculate_request_cost") or _get_setting(
            "calculate_request_cost", calculate_request_cost
        )
        usage = result.usage or {}
        req_cost = cost_calc_fn(
            model=result.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        store_fn = _get_setting("_get_local_cost_store") or _get_setting(
            "get_local_cost_store", get_local_cost_store
        )
        store = store_fn()
        store.record_request(req_cost, client_key=key)
        response_headers.update(req_cost.to_headers())

    # Wenn Streaming gewünscht: Antwort als SSE emulieren
    if want_stream:
        stream_delay_ms = _get_setting("STREAM_CHUNK_DELAY_MS", STREAM_CHUNK_DELAY_MS)

        def sse_generator():
            if result.tool_calls_data:
                # Tool-Call-Chunks im OpenAI-Streaming-Format
                for chunk in build_tool_calls_sse_chunks(result.completion_id, result.created_ts, result.model, result.tool_calls_data):
                    if stream_delay_ms > 0:
                        time.sleep(stream_delay_ms / 1000.0)
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Normaler Text-Response als SSE
                # Chunk 1: role delta
                if stream_delay_ms > 0:
                    time.sleep(stream_delay_ms / 1000.0)
                yield f"data: {json.dumps({'id': result.completion_id, 'object': 'chat.completion.chunk', 'created': result.created_ts, 'model': result.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
                # Chunk 2: Content
                if stream_delay_ms > 0:
                    time.sleep(stream_delay_ms / 1000.0)
                yield f"data: {json.dumps({'id': result.completion_id, 'object': 'chat.completion.chunk', 'created': result.created_ts, 'model': result.model, 'choices': [{'index': 0, 'delta': {'content': result.content}, 'finish_reason': None}]})}\n\n"
                # Chunk 3: finish
                if stream_delay_ms > 0:
                    time.sleep(stream_delay_ms / 1000.0)
                yield f"data: {json.dumps({'id': result.completion_id, 'object': 'chat.completion.chunk', 'created': result.created_ts, 'model': result.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': result.finish_reason}], 'usage': result.usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream", headers=response_headers)

    # Kein Streaming: normaler JSON-Response
    if result.tool_calls_data:
        return JSONResponse(
            content=build_tool_calls_response(result.completion_id, result.created_ts, result.model, result.tool_calls_data, result.usage),
            headers=response_headers,
        )

    return JSONResponse(
        content={
            "id": result.completion_id,
            "object": "chat.completion",
            "created": result.created_ts,
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.content},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": result.usage,
        },
        headers=response_headers,
    )


async def responses(request: Request, key: str = Depends(verify_key)):
    """
    OpenAI Responses API Endpunkt (POST /v1/responses) für OpenAI Codex CLI,
    Codex Desktop App und moderne Agent-Harnesses.
    """
    try:
        body = await request.json()
    except Exception:
        log.warning("Responses request rejected (400): invalid JSON body")
        raise HTTPException(status_code=400, detail="invalid JSON body")

    max_req_chars = _get_setting("MAX_REQUEST_JSON_CHARS", MAX_REQUEST_JSON_CHARS)
    size_validator = _get_setting("validate_request_json_size", validate_request_json_size)
    size_validator(body, max_chars=max_req_chars)

    body_validator = _get_setting("_validate_responses_request_body") or _get_setting(
        "validate_responses_request_body", validate_responses_request_body
    )
    body_validator(body)

    rate_limiter = _get_setting("_enforce_chat_rate_limit") or _get_setting(
        "enforce_chat_rate_limit", enforce_chat_rate_limit
    )
    rate_limiter(request, key)

    debug_dumps = _get_setting("DEBUG_DUMPS", False)
    if debug_dumps:
        _dump_path = Path(__file__).resolve().parent.parent / "last_responses_request.json"
        try:
            with open(_dump_path, "w", encoding="utf-8") as _f:
                json.dump(redact_sensitive(body), _f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    normalizer = _get_setting("normalize_responses_request", normalize_responses_request)
    model, messages, tools, tool_choice, optional = normalizer(body)
    want_stream = bool(body.get("stream"))

    if "extra_body" in body and "tailoredAiId" in body["extra_body"]:
        optional["extra_body"] = {"tailoredAiId": body["extra_body"]["tailoredAiId"]}

    cost_cache_fn = _get_setting("_get_cost_cache_with_lazy_refresh") or _get_setting(
        "get_cost_cache_with_lazy_refresh", get_cost_cache_with_lazy_refresh
    )
    cost_cache = cost_cache_fn()
    cost_headers_fn = _get_setting("_build_cost_headers") or _get_setting("build_cost_headers", build_cost_headers)
    response_headers = cost_headers_fn(cost_cache)

    try:
        result = await _execute_completion_pipeline(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            optional_params=optional,
            stream=want_stream,
        )
    except (HTTPException, AcademicAIError):
        raise
    except Exception as e:
        log.error(f"Responses completion failed: model={model} error={e}")
        raise HTTPException(status_code=502, detail=str(e))

    # Lokale Kostenberechnung & Header
    enable_local_cost = bool(_get_setting("ENABLE_LOCAL_COST_TRACKING", ENABLE_LOCAL_COST_TRACKING))
    if enable_local_cost:
        cost_calc_fn = _get_setting("_calculate_request_cost") or _get_setting(
            "calculate_request_cost", calculate_request_cost
        )
        usage = result.usage or {}
        req_cost = cost_calc_fn(
            model=result.model,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
        store_fn = _get_setting("_get_local_cost_store") or _get_setting(
            "get_local_cost_store", get_local_cost_store
        )
        store = store_fn()
        store.record_request(req_cost, client_key=key)
        response_headers.update(req_cost.to_headers())

    if want_stream:
        stream_delay_ms = _get_setting("STREAM_CHUNK_DELAY_MS", STREAM_CHUNK_DELAY_MS)
        sse_builder = _get_setting("build_responses_sse_events", build_responses_sse_events)
        return StreamingResponse(
            sse_builder(
                completion_id=result.completion_id,
                created_ts=result.created_ts,
                model=result.model,
                content=result.content,
                tool_calls_data=result.tool_calls_data,
                usage=result.usage,
                delay_ms=stream_delay_ms,
            ),
            media_type="text/event-stream",
            headers=response_headers,
        )

    output_builder = _get_setting("build_responses_output", build_responses_output)
    return JSONResponse(
        content=output_builder(
            completion_id=result.completion_id,
            created_ts=result.created_ts,
            model=result.model,
            content=result.content,
            tool_calls_data=result.tool_calls_data,
            usage=result.usage,
        ),
        headers=response_headers,
    )


# --- App Factory ---


def create_app() -> FastAPI:
    """
    Erstellt und konfiguriert die FastAPI-Instanz mit modernem Lifespan-Handler
    und registrierten Endpunkten.
    """
    application = FastAPI(
        title="AcademicAI Proxy",
        description="OpenAI-kompatibler Proxy für AcademicAI",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(AcademicAIError)
    async def academicai_error_handler(request: Request, exc: AcademicAIError):
        headers = None
        try:
            cache_fn = _get_setting("_get_cost_cache_with_lazy_refresh") or _get_setting(
                "get_cost_cache_with_lazy_refresh", get_cost_cache_with_lazy_refresh
            )
            cost_headers_fn = _get_setting("_build_cost_headers") or _get_setting(
                "build_cost_headers", build_cost_headers
            )
            cache = cache_fn()
            headers = cost_headers_fn(cache)
        except Exception:
            pass

        return JSONResponse(
            status_code=exc.status_code if exc.status_code else 502,
            content={
                "error": {
                    "message": exc.message,
                    "type": exc.error_type,
                    "param": exc.param,
                    "code": exc.error_code,
                }
            },
            headers=headers,
        )

    application.get("/health")(health)
    application.get("/internal/cost-status")(cost_status)
    application.get("/v1/models")(list_models)
    application.post("/v1/chat/completions")(chat_completions)
    application.post("/v1/responses")(responses)

    return application


app = create_app()


__all__ = [
    "create_app",
    "app",
    "lifespan",
    "security",
    "verify_key",
    "health",
    "cost_status",
    "list_models",
    "chat_completions",
    "responses",
    "CanonicalResult",
]