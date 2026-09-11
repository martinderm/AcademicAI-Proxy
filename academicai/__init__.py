"""
AcademicAI — OpenAI-kompatibler Client ohne LiteLLM.

Schnellstart:
    from academicai import completion, get_models

    models = get_models()
    response = completion(model="gpt-4o", messages=[{"role": "user", "content": "Hallo!"}])
    print(response.choices[0].message.content)
"""

from .provider import AcademicAIProvider, CompletionResponse

_provider: AcademicAIProvider | None = None


def _get_provider() -> AcademicAIProvider:
    global _provider
    if _provider is None:
        _provider = AcademicAIProvider()
    return _provider


def completion(model: str, messages: list, **kwargs) -> CompletionResponse:
    """
    AcademicAI Chat Completion.

    Args:
        model:    Modellname, z.B. "gpt-4o", "gpt-5", "Mistral-Large-3"
        messages: Liste von {"role": ..., "content": ...}
        **kwargs: Optional: temperature, max_tokens, max_completion_tokens,
                  frequency_penalty, presence_penalty, response_format,
                  reasoning_effort, verbosity, seed, stop,
                  extra_body={"tailoredAiId": "..."} für RAG

    Returns:
        CompletionResponse mit .choices[0].message.content etc.
    """
    return _get_provider().completion(model=model, messages=messages, optional_params=kwargs)


def get_models() -> dict:
    """
    Listet verfügbare AcademicAI-Modelle.
    Gibt OpenAI-kompatibles /v1/models Format zurück.
    """
    return _get_provider().get_models()


from .config import Settings, get_settings, validate_config  # noqa: F401
from .request_guards import (  # noqa: F401
    validate_chat_request_body,
    _validate_chat_request_body,
    validate_responses_request_body,
    _validate_responses_request_body,
    validate_request_json_size,
    _validate_request_json_size,
    rate_limit_bucket,
    _rate_limit_bucket,
    prune_rate_limit_buckets,
    _prune_rate_limit_buckets,
    enforce_chat_rate_limit,
    _enforce_chat_rate_limit,
)
from .responses import (  # noqa: F401
    normalize_responses_request,
    build_responses_output,
    build_responses_sse_events,
)
from .tool_emulation import (  # noqa: F401
    inject_tools_into_messages,
    parse_tool_call,
    strip_tool_call_tag,
    build_tool_calls_response,
    build_tool_calls_sse_chunks,
    apply_post_tool_guard,
    _apply_post_tool_guard,
)
from .transformation import (  # noqa: F401
    extract_text_content,
    _extract_text_content,
)
from .cost_monitoring import (  # noqa: F401
    read_cost_cache,
    _read_cost_cache,
    write_cost_cache,
    _write_cost_cache,
    is_cost_cache_stale,
    _is_cost_cache_stale,
    build_cost_headers,
    _build_cost_headers,
    fetch_cost_snapshot,
    _fetch_cost_snapshot,
    refresh_cost_cache_sync,
    _refresh_cost_cache_sync,
    refresh_cost_cache_background,
    _refresh_cost_cache_background,
    get_cost_cache_with_lazy_refresh,
    _get_cost_cache_with_lazy_refresh,
    get_cost_status_payload,
)
from .runtime import (  # noqa: F401
    write_pid_file,
    _write_pid_file,
    cleanup_pid_file,
    _cleanup_pid_file,
    check_backend_health,
    _check_backend_health,
    get_health_payload,
)
from .logging_config import (  # noqa: F401
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
from .humanization import (  # noqa: F401
    build_humanization_messages,
    _build_humanization_messages,
    is_human_readable_target,
    _is_human_readable_target,
    last_user_text,
    _last_user_text,
    run_humanization_pass,
    _run_humanization_pass,
)

__all__ = [
    "AcademicAIProvider", "CompletionResponse", "completion", "get_models",
    "Settings", "get_settings", "validate_config",
    "validate_chat_request_body", "_validate_chat_request_body",
    "validate_responses_request_body", "_validate_responses_request_body",
    "normalize_responses_request", "build_responses_output", "build_responses_sse_events",
    "validate_request_json_size", "_validate_request_json_size",
    "rate_limit_bucket", "_rate_limit_bucket",
    "prune_rate_limit_buckets", "_prune_rate_limit_buckets",
    "enforce_chat_rate_limit", "_enforce_chat_rate_limit",
    "inject_tools_into_messages", "parse_tool_call", "strip_tool_call_tag",
    "build_tool_calls_response", "build_tool_calls_sse_chunks",
    "apply_post_tool_guard", "_apply_post_tool_guard",
    "extract_text_content", "_extract_text_content",
    "read_cost_cache", "_read_cost_cache",
    "write_cost_cache", "_write_cost_cache",
    "is_cost_cache_stale", "_is_cost_cache_stale",
    "build_cost_headers", "_build_cost_headers",
    "fetch_cost_snapshot", "_fetch_cost_snapshot",
    "refresh_cost_cache_sync", "_refresh_cost_cache_sync",
    "refresh_cost_cache_background", "_refresh_cost_cache_background",
    "get_cost_cache_with_lazy_refresh", "_get_cost_cache_with_lazy_refresh",
    "get_cost_status_payload",
    "write_pid_file", "_write_pid_file",
    "cleanup_pid_file", "_cleanup_pid_file",
    "check_backend_health", "_check_backend_health",
    "get_health_payload",
    "configure_logging", "get_logger", "close_handlers",
    "log", "log_formatter", "info_handler", "error_handler",
    "console_handler", "root_logger",
    "build_humanization_messages", "_build_humanization_messages",
    "is_human_readable_target", "_is_human_readable_target",
    "last_user_text", "_last_user_text",
    "run_humanization_pass", "_run_humanization_pass",
]


