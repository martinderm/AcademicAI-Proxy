"""
Unit tests for academicai.app module (FastAPI application factory & lifespan).

Validates:
- create_app() instantiates a FastAPI app with required routes (/health, /internal/cost-status, /v1/models, /v1/chat/completions)
- Lifespan context manager executes startup (validate_config, write_pid_file) and shutdown (cleanup_pid_file) hooks
- Lifespan execution via TestClient context manager
- Route handler direct execution and delegation: health, cost_status, list_models, verify_key
- Error handling in list_models (502 on failure)
- Tool post-guard in academicai.tool_emulation (apply_post_tool_guard and _apply_post_tool_guard)
- Backward compatibility re-exports on server and academicai packages
"""

import asyncio
import inspect
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

import academicai
import academicai.app as ai_app
from academicai.app import (
    create_app,
    app,
    lifespan,
    verify_key,
    health,
    cost_status,
    list_models,
    chat_completions,
)
from academicai.tool_emulation import (
    apply_post_tool_guard,
    _apply_post_tool_guard,
)
from academicai.transformation import (
    extract_text_content,
    _extract_text_content,
)
import server


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. create_app Instantiation & Route Registration
# ---------------------------------------------------------------------------


def test_create_app_instantiates_fastapi_with_routes():
    test_app = create_app()
    assert isinstance(test_app, FastAPI)
    assert test_app.title == "AcademicAI Proxy"

    registered = {}
    for route in test_app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            registered[route.path] = route.methods

    assert "/health" in registered
    assert "GET" in registered["/health"]

    assert "/internal/cost-status" in registered
    assert "GET" in registered["/internal/cost-status"]

    assert "/v1/models" in registered
    assert "GET" in registered["/v1/models"]

    assert "/v1/chat/completions" in registered
    assert "POST" in registered["/v1/chat/completions"]

    assert "/v1/responses" in registered
    assert "POST" in registered["/v1/responses"]


def test_module_level_app_instance():
    assert isinstance(app, FastAPI)
    assert app.title == "AcademicAI Proxy"


# ---------------------------------------------------------------------------
# 2. Modern Lifespan Handler Execution
# ---------------------------------------------------------------------------


def test_lifespan_lifecycle_direct(monkeypatch):
    events = []
    monkeypatch.setattr(ai_app, "validate_config", lambda: events.append("validate_config"))
    monkeypatch.setattr(ai_app, "write_pid_file", lambda: events.append("write_pid_file"))
    monkeypatch.setattr(ai_app, "cleanup_pid_file", lambda: events.append("cleanup_pid_file"))

    async def _test():
        test_app = create_app()
        async with lifespan(test_app):
            assert events == ["validate_config", "write_pid_file"]
        assert events == ["validate_config", "write_pid_file", "cleanup_pid_file"]

    _run(_test())


def test_lifespan_via_test_client(monkeypatch):
    events = []
    monkeypatch.setattr(ai_app, "validate_config", lambda: events.append("start_config"))
    monkeypatch.setattr(ai_app, "write_pid_file", lambda: events.append("start_pid"))
    monkeypatch.setattr(ai_app, "cleanup_pid_file", lambda: events.append("stop_pid"))

    test_app = create_app()
    with TestClient(test_app):
        assert "start_config" in events
        assert "start_pid" in events
        assert "stop_pid" not in events
    assert "stop_pid" in events


# ---------------------------------------------------------------------------
# 3. Route Handlers Delegation & Unit Behavior
# ---------------------------------------------------------------------------


def test_health_handler_delegation(monkeypatch):
    monkeypatch.setattr(ai_app, "check_backend_health", lambda: {"enabled": True, "ok": True, "latency_ms": 7})
    payload = health()
    assert isinstance(payload, dict)
    assert payload.get("status") == "ok"
    assert payload.get("service") == "academicai-proxy"
    assert payload.get("backend", {}).get("ok") is True


def test_cost_status_handler_delegation(monkeypatch):
    mock_cache = {
        "total_cost": 15.25,
        "total_clients": 2,
        "cost_entries": 4,
        "updated_at": "2026-09-06T12:00:00+00:00",
        "source": "cache",
    }
    monkeypatch.setattr(ai_app, "get_cost_cache_with_lazy_refresh", lambda: mock_cache)
    payload = cost_status(key="valid-key")
    assert isinstance(payload, dict)
    assert payload.get("total_cost") == 15.25
    assert payload.get("total_clients") == 2
    assert payload.get("source") == "cache"
    assert "local_cost_tracking" in payload
    assert "model_catalog" in payload["local_cost_tracking"]
    assert "pricing_cache" in payload["local_cost_tracking"]


def test_list_models_handler_success(monkeypatch):
    mock_models = {
        "object": "list",
        "data": [{"id": "gpt-4o", "object": "model", "owned_by": "academicai"}],
    }
    monkeypatch.setattr(academicai, "get_models", lambda: mock_models)
    res = _run(list_models(key="valid-key"))
    assert res == mock_models


def test_list_models_handler_error_raises_502(monkeypatch):
    def _fail():
        raise RuntimeError("backend unreachable")

    monkeypatch.setattr(academicai, "get_models", _fail)
    with pytest.raises(HTTPException) as exc_info:
        _run(list_models(key="valid-key"))
    assert exc_info.value.status_code == 502
    assert "backend unreachable" in exc_info.value.detail


def test_verify_key_behavior(monkeypatch):
    monkeypatch.setattr(ai_app, "_get_setting", lambda name, default=None: "secret-key" if name == "API_KEY" else default)

    # Success case
    creds_ok = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-key")
    assert verify_key(creds_ok) == "secret-key"

    # Mismatch case
    creds_bad = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-key")
    with pytest.raises(HTTPException) as exc_bad:
        verify_key(creds_bad)
    assert exc_bad.value.status_code == 401

    # Missing credentials case
    with pytest.raises(HTTPException) as exc_none:
        verify_key(None)
    assert exc_none.value.status_code == 401


# ---------------------------------------------------------------------------
# 4. apply_post_tool_guard from academicai.tool_emulation
# ---------------------------------------------------------------------------


def test_apply_post_tool_guard_alias_and_cases():
    assert _apply_post_tool_guard is apply_post_tool_guard

    # Case 1: has_tools is False -> unchanged
    msgs = [{"role": "tool", "content": "OK"}]
    assert apply_post_tool_guard(msgs, has_tools=False) == msgs

    # Case 2: empty messages -> unchanged
    assert apply_post_tool_guard([], has_tools=True) == []

    # Case 3: last message not role=tool -> unchanged
    msgs_user = [{"role": "user", "content": "Hello"}]
    assert apply_post_tool_guard(msgs_user, has_tools=True) == msgs_user

    # Case 4: last message tool without error -> NO_FURTHER_TOOL_CALLS
    msgs_ok = [
        {"role": "user", "content": "Find info"},
        {"role": "tool", "content": "File contents here"},
    ]
    guarded_ok = apply_post_tool_guard(msgs_ok, has_tools=True)
    assert len(guarded_ok) == 3
    assert guarded_ok[0]["role"] == "system"
    assert "NO_FURTHER_TOOL_CALLS" in guarded_ok[0]["content"]

    # Case 5: last message tool with error -> TOOL_RESULT_ERROR
    msgs_err = [
        {"role": "user", "content": "Find info"},
        {"role": "tool", "content": "Error: File not found exception"},
    ]
    guarded_err = apply_post_tool_guard(msgs_err, has_tools=True)
    assert len(guarded_err) == 3
    assert guarded_err[0]["role"] == "system"
    assert "TOOL_RESULT_ERROR" in guarded_err[0]["content"]


# ---------------------------------------------------------------------------
# 5. Re-exports and Aliases Verification
# ---------------------------------------------------------------------------


def test_backward_compatibility_server_and_academicai_reexports():
    server_symbols = [
        "create_app",
        "app",
        "lifespan",
        "verify_key",
        "health",
        "cost_status",
        "list_models",
        "chat_completions",
        "apply_post_tool_guard",
        "_apply_post_tool_guard",
        "extract_text_content",
        "_extract_text_content",
        "AcademicAIError",
        "QuotaExceededError",
        "map_error",
    ]

    for sym in server_symbols:
        assert hasattr(server, sym), f"server is missing expected re-export: {sym}"

    package_symbols = [
        "apply_post_tool_guard",
        "_apply_post_tool_guard",
        "extract_text_content",
        "_extract_text_content",
        "AcademicAIError",
        "QuotaExceededError",
        "map_error",
    ]

    for sym in package_symbols:
        assert hasattr(academicai, sym), f"academicai is missing expected re-export: {sym}"

    app_module_symbols = [
        "create_app",
        "app",
        "lifespan",
        "verify_key",
        "health",
        "cost_status",
        "list_models",
        "chat_completions",
        "responses",
    ]

    for sym in app_module_symbols:
        assert hasattr(ai_app, sym), f"academicai.app is missing expected symbol: {sym}"


def test_isolated_module_imports_without_server():
    """Verify that every academicai domain module can be imported independently without server.py."""
    import subprocess
    import sys

    modules = [
        "academicai.config",
        "academicai.auth",
        "academicai.security",
        "academicai.transformation",
        "academicai.request_guards",
        "academicai.tool_emulation",
        "academicai.cost_monitoring",
        "academicai.runtime",
        "academicai.logging_config",
        "academicai.humanization",
        "academicai.responses",
        "academicai.errors",
        "academicai.app",
        "academicai",
    ]

    for mod in modules:
        cmd = [sys.executable, "-c", f"import sys; import {mod}; assert 'server' not in sys.modules"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"Module {mod} failed isolated import test: {res.stderr}"


# ---------------------------------------------------------------------------
# 5. Error Mapping & OpenAI Format Exception Handler Tests
# ---------------------------------------------------------------------------


def test_quota_exceeded_error_defaults():
    from academicai.errors import QuotaExceededError
    err = QuotaExceededError("Budget exhausted")
    assert err.status_code == 429
    assert err.error_code == "insufficient_quota"
    assert err.error_type == "insufficient_quota"
    assert err.message == "Budget exhausted"


def test_map_error_internal_error_code_201():
    from academicai.errors import map_error, QuotaExceededError
    upstream_payload = {
        "message": "API request failed",
        "meta": {
            "error": {
                "internalErrorCode": 201,
                "message": "API Client Error: Cost limit reached",
            }
        },
    }
    err = map_error(403, upstream_payload)
    assert isinstance(err, QuotaExceededError)
    assert err.status_code == 429
    assert err.error_code == "insufficient_quota"
    assert err.error_type == "insufficient_quota"
    assert "Cost limit reached" in err.message


def test_map_error_cost_limit_string_detection():
    from academicai.errors import map_error, QuotaExceededError
    err = map_error(400, {"message": "User cost limit exceeded for this billing period"})
    assert isinstance(err, QuotaExceededError)
    assert err.status_code == 429


def test_map_error_status_429():
    from academicai.errors import map_error, QuotaExceededError
    err = map_error(429, {"message": "Too many requests"})
    assert isinstance(err, QuotaExceededError)
    assert err.status_code == 429


def test_map_error_kb_unavailable():
    from academicai.errors import map_error, ServiceUnavailableError
    err = map_error(500, {"code": "KB_UNAVAILABLE", "message": "Knowledge Base is down"})
    assert isinstance(err, ServiceUnavailableError)
    assert err.error_code == "kb_unavailable"


def test_map_error_standard_status_codes():
    from academicai.errors import (
        map_error,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        BadRequestError,
        ServiceUnavailableError,
    )
    assert isinstance(map_error(401, {"message": "Invalid key"}), AuthenticationError)
    assert isinstance(map_error(403, {"message": "Forbidden"}), PermissionDeniedError)
    assert isinstance(map_error(404, {"message": "Not found"}), NotFoundError)
    assert isinstance(map_error(422, {"message": "Validation failed"}), BadRequestError)
    assert isinstance(map_error(503, {"message": "Service unavailable"}), ServiceUnavailableError)


def test_chat_completion_quota_exceeded_returns_429_openai_format(monkeypatch):
    from academicai.errors import QuotaExceededError
    from fastapi.testclient import TestClient
    import server

    def _mock_completion(*args, **kwargs):
        raise QuotaExceededError("AcademicAI Cost Limit Reached: API Client Error: Cost limit reached")

    monkeypatch.setattr(server.academicai, "completion", _mock_completion)
    client = TestClient(server.app)

    payload = {"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hello"}]}
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {server.API_KEY}"},
        json=payload,
    )

    assert resp.status_code == 429
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "insufficient_quota"
    assert data["error"]["type"] == "insufficient_quota"
    assert "Cost limit reached" in data["error"]["message"]
    assert data["error"]["param"] is None


def test_responses_api_quota_exceeded_returns_429_openai_format(monkeypatch):
    from academicai.errors import QuotaExceededError
    from fastapi.testclient import TestClient
    import server

    def _mock_completion(*args, **kwargs):
        raise QuotaExceededError("AcademicAI Cost Limit Reached: API Client Error: Cost limit reached")

    monkeypatch.setattr(server.academicai, "completion", _mock_completion)
    client = TestClient(server.app)

    payload = {"model": "gpt-5-mini", "input": "hello"}
    resp = client.post(
        "/v1/responses",
        headers={"Authorization": f"Bearer {server.API_KEY}"},
        json=payload,
    )

    assert resp.status_code == 429
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "insufficient_quota"
    assert data["error"]["type"] == "insufficient_quota"
    assert "Cost limit reached" in data["error"]["message"]


def test_list_models_quota_exceeded_returns_429_openai_format(monkeypatch):
    from academicai.errors import QuotaExceededError
    from fastapi.testclient import TestClient
    import server

    def _mock_get_models():
        raise QuotaExceededError("AcademicAI Cost Limit Reached: API Client Error: Cost limit reached")

    monkeypatch.setattr(server.academicai, "get_models", _mock_get_models)
    client = TestClient(server.app)

    resp = client.get(
        "/v1/models",
        headers={"Authorization": f"Bearer {server.API_KEY}"},
    )

    assert resp.status_code == 429
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "insufficient_quota"