import json
import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("ACADEMICAI_PROXY_API_KEY", "test-proxy-key-123456")
os.environ.setdefault("ACADEMICAI_HEALTH_CHECK_BACKEND", "false")

import server
from academicai.provider import AcademicAIProvider


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {
            "data": {
                "content": "ok",
                "finishReason": "stop",
                "usage": {"promptTokens": 1, "completionTokens": 1, "totalTokens": 2},
            }
        }
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        return _FakeResponse()


def _mock_completion(*args, **kwargs):
    msg = SimpleNamespace(role="assistant", content="Hallo")
    choice = SimpleNamespace(index=0, message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    return SimpleNamespace(
        id="cmpl-test",
        created=123,
        model="gpt-5-mini",
        choices=[choice],
        usage=usage,
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server.academicai, "completion", _mock_completion)
    monkeypatch.setattr(server, "_check_backend_health", lambda: {"enabled": True, "ok": True})
    return TestClient(server.app)


def _auth_headers():
    return {"Authorization": f"Bearer {server.API_KEY}"}


def test_validate_proxy_api_key_rejects_placeholder():
    with pytest.raises(RuntimeError):
        server._validate_proxy_api_key("academicai-proxy")


def test_validate_proxy_api_key_rejects_short_key():
    with pytest.raises(RuntimeError):
        server._validate_proxy_api_key("short-key")


def test_chat_requires_auth(client):
    resp = client.post("/v1/chat/completions", json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 401


def test_chat_invalid_messages_type_returns_422(client):
    resp = client.post(
        "/v1/chat/completions",
        headers=_auth_headers(),
        json={"model": "gpt-5-mini", "messages": "invalid"},
    )
    assert resp.status_code == 422
    assert "messages" in resp.text


def test_chat_invalid_json_returns_400(client):
    resp = client.post(
        "/v1/chat/completions",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        data="{",
    )
    assert resp.status_code == 400


def test_chat_message_size_limit_returns_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_MESSAGE_TEXT_CHARS", 4)
    resp = client.post(
        "/v1/chat/completions",
        headers=_auth_headers(),
        json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 413
    assert "content exceeds limit" in resp.text


def test_chat_tools_count_limit_returns_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_TOOLS", 2)
    tools = [
        {"type": "function", "function": {"name": f"tool_{i}", "description": "test"}}
        for i in range(3)
    ]
    resp = client.post(
        "/v1/chat/completions",
        headers=_auth_headers(),
        json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}], "tools": tools},
    )
    assert resp.status_code == 413
    assert "tools exceed limit" in resp.text


def test_chat_tool_schema_size_limit_returns_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_TOOL_SCHEMA_CHARS", 50)
    tools = [
        {"type": "function", "function": {"name": "oversized_tool", "description": "x" * 100}}
    ]
    resp = client.post(
        "/v1/chat/completions",
        headers=_auth_headers(),
        json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}], "tools": tools},
    )
    assert resp.status_code == 413
    assert "exceeds limit" in resp.text


def test_chat_request_body_size_limit_returns_413(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_REQUEST_JSON_CHARS", 80)
    resp = client.post(
        "/v1/chat/completions",
        headers=_auth_headers(),
        json={"model": "gpt-5-mini", "messages": [{"role": "user", "content": "this payload will exceed eighty chars easily"}]},
    )
    assert resp.status_code == 413
    assert "request body exceeds limit" in resp.text


def test_chat_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MINUTE", 1)
    server._rate_limit_buckets.clear()

    payload = {"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]}
    first = client.post("/v1/chat/completions", headers=_auth_headers(), json=payload)
    second = client.post("/v1/chat/completions", headers=_auth_headers(), json=payload)

    assert first.status_code == 200
    assert second.status_code == 429


def test_chat_completion_exception_returns_502(client, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("backend down")

    monkeypatch.setattr(server.academicai, "completion", _boom)
    payload = {"model": "gpt-5-mini", "messages": [{"role": "user", "content": "hi"}]}
    resp = client.post("/v1/chat/completions", headers=_auth_headers(), json=payload)

    assert resp.status_code == 502


def test_chat_streaming_returns_done_marker(client):
    payload = {
        "model": "gpt-5-mini",
        "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }
    with client.stream("POST", "/v1/chat/completions", headers=_auth_headers(), json=payload) as resp:
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    assert resp.status_code == 200
    assert "data: [DONE]" in body


def test_health_degraded_when_backend_unhealthy(client, monkeypatch):
    monkeypatch.setattr(server, "_check_backend_health", lambda: {"enabled": True, "ok": False, "error": "timeout"})
    resp = client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["backend"]["ok"] is False


def test_provider_debug_dump_redacts_secrets(monkeypatch):
    captured = {}

    def _capture_dump(obj, fp, ensure_ascii=False, indent=2):
        captured["obj"] = obj

    monkeypatch.setattr("academicai.provider.DEBUG_DUMPS", True)
    monkeypatch.setattr("academicai.provider.get_base_url", lambda: "https://example.test")
    monkeypatch.setattr(
        "academicai.provider.get_headers",
        lambda: {"X-Client-ID": "id-123", "X-Client-Secret": "very-secret"},
    )
    monkeypatch.setattr("academicai.provider.httpx.Client", _FakeClient)
    monkeypatch.setattr("academicai.provider.json.dump", _capture_dump)

    provider = AcademicAIProvider()
    provider.completion(model="gpt-5-mini", messages=[{"role": "user", "content": "token abc"}])

    debug_obj = captured["obj"]
    assert debug_obj["headers"]["X-Client-Secret"] == "***REDACTED***"
    assert "very-secret" not in json.dumps(debug_obj)
