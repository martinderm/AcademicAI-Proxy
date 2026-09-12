"""
AcademicAI OpenAI-kompatibler Proxy Server

Exponiert AcademicAI als OpenAI-kompatible API auf Port 11435.
OpenClaw (und andere Tools) können ihn wie jeden OpenAI-kompatiblen Provider nutzen.

Start:
    py server.py

Endpoints:
    GET  /v1/models                 → Modell-Liste
    POST /v1/chat/completions       → Chat Completion (inkl. Streaming-Emulation)
    POST /v1/responses              → OpenAI Responses API (Codex CLI / Desktop)
    GET  /health                    → Health Check
    GET  /internal/cost-status      → Interner Kosten-Status
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import uvicorn
import academicai
from academicai.auth import get_base_url, get_headers
from academicai.security import redact_sensitive
from academicai.errors import (
    AcademicAIError,
    AuthenticationError,
    PermissionDeniedError,
    QuotaExceededError,
    NotFoundError,
    BadRequestError,
    ServiceUnavailableError,
    map_error,
)

from academicai.transformation import (
    extract_text_content,
    _extract_text_content,
)

from academicai.tool_emulation import (
    apply_post_tool_guard,
    _apply_post_tool_guard,
    inject_tools_into_messages,
    parse_tool_calls,
    extract_respond_content,
    format_arbitrary_json_as_codeblock,
    format_arbitrary_json_for_humans,
    build_tool_calls_response,
    build_tool_calls_sse_chunks,
)

from academicai.request_guards import (
    validate_request_json_size,
    _validate_request_json_size,
    validate_chat_request_body,
    _validate_chat_request_body,
    validate_responses_request_body,
    _validate_responses_request_body,
    rate_limit_bucket,
    _rate_limit_bucket,
    prune_rate_limit_buckets,
    _prune_rate_limit_buckets,
    enforce_chat_rate_limit,
    _enforce_chat_rate_limit,
    _rate_limit_buckets,
    _rate_limit_lock,
)

from academicai.responses import (
    normalize_responses_request,
    build_responses_output,
    build_responses_sse_events,
)

from academicai.humanization import (
    build_humanization_messages,
    _build_humanization_messages,
    is_human_readable_target,
    _is_human_readable_target,
    last_user_text,
    _last_user_text,
    run_humanization_pass,
    _run_humanization_pass,
)

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

from academicai.cost_monitoring import (
    _cost_lock,
    _cost_refresh_in_flight,
    _now_utc_iso,
    _parse_iso_ts,
    _safe_float,
    _extract_cost_summary,
    _read_cost_cache,
    _write_cost_cache,
    _is_cost_cache_stale,
    _build_cost_headers,
    _fetch_cost_snapshot,
    _refresh_cost_cache_sync,
    _refresh_cost_cache_background,
    _get_cost_cache_with_lazy_refresh,
    read_cost_cache,
    write_cost_cache,
    is_cost_cache_stale,
    build_cost_headers,
    fetch_cost_snapshot,
    refresh_cost_cache_sync,
    refresh_cost_cache_background,
    get_cost_cache_with_lazy_refresh,
    get_cost_status_payload,
)

from academicai.runtime import (
    write_pid_file,
    _write_pid_file,
    cleanup_pid_file,
    _cleanup_pid_file,
    check_backend_health,
    _check_backend_health,
    get_health_payload,
)

from academicai.logging_config import (
    configure_logging,
    get_logger,
    close_handlers,
    log,
    log_formatter,
    info_handler,
    error_handler,
    console_handler,
    root_logger,
)

log_file_path = LOG_FILE_PATH
err_file_path = ERR_FILE_PATH

# --- Application Factory & Endpoints ---
from academicai.app import (
    create_app,
    app,
    lifespan,
    security,
    verify_key,
    health,
    cost_status,
    list_models,
    chat_completions,
    responses,
)


def _on_startup() -> None:
    validate_config()
    _write_pid_file()


def _on_shutdown() -> None:
    _cleanup_pid_file()


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