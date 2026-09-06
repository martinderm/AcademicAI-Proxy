"""
Characterization tests for AcademicAI-Proxy public endpoints.
Captures and protects existing public contracts before modularization refactoring:
  - GET /health
  - GET /internal/cost-status
  - GET /v1/models
  - POST /v1/chat/completions
"""

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture
def client():
    return TestClient(server.app)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {server.API_KEY}"}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_ok_structure(client, monkeypatch):
    """GET /health returns 200 OK and expected structure when backend is healthy."""
    monkeypatch.setattr(server, "_check_backend_health", lambda: {"enabled": True, "ok": True, "latency_ms": 5})
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "academicai-proxy"
    assert "backend" in data
    assert data["backend"]["enabled"] is True
    assert data["backend"]["ok"] is True


def test_health_endpoint_returns_degraded_when_backend_unhealthy(client, monkeypatch):
    """GET /health returns 200 OK with status degraded when backend is unhealthy."""
    monkeypatch.setattr(server, "_check_backend_health", lambda: {"enabled": True, "ok": False, "latency_ms": 10})
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "degraded"
    assert data.get("service") == "academicai-proxy"
    assert data["backend"]["ok"] is False


# ---------------------------------------------------------------------------
# GET /internal/cost-status
# ---------------------------------------------------------------------------


def test_cost_status_unauthenticated_returns_401(client):
    """GET /internal/cost-status returns 401 Unauthorized without auth headers."""
    resp = client.get("/internal/cost-status")
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Unauthorized"


def test_cost_status_authenticated_returns_expected_structure(client, monkeypatch):
    """GET /internal/cost-status returns 200 OK with expected fields when authenticated."""
    monkeypatch.setattr(
        server,
        "_get_cost_cache_with_lazy_refresh",
        lambda: {
            "total_cost": 42.50,
            "total_clients": 3,
            "cost_entries": 10,
            "updated_at": "2026-09-06T12:00:00+00:00",
            "source": "cache",
        },
    )
    resp = client.get("/internal/cost-status", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert "enabled" in data
    assert data["total_cost"] == 42.50
    assert data["total_clients"] == 3
    assert data["cost_entries"] == 10
    assert data["updated_at"] == "2026-09-06T12:00:00+00:00"
    assert "is_stale" in data
    assert data["source"] == "cache"


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


def test_models_unauthenticated_returns_401(client):
    """GET /v1/models returns 401 Unauthorized without auth headers."""
    resp = client.get("/v1/models")
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Unauthorized"


def test_models_authenticated_returns_list_matching_allowed_models(client, monkeypatch):
    """GET /v1/models returns 200 OK with object=list and data matching server.ALLOWED_MODELS."""
    mock_models_response = {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "academicai"}
            for m in server.ALLOWED_MODELS
        ],
    }
    monkeypatch.setattr(server.academicai, "get_models", lambda: mock_models_response)

    resp = client.get("/v1/models", headers=_auth_headers())
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("object") == "list"
    assert isinstance(data.get("data"), list)
    model_ids = [item["id"] for item in data["data"]]
    assert model_ids == server.ALLOWED_MODELS
    for item in data["data"]:
        assert item.get("object") == "model"
        assert item.get("owned_by") == "academicai"


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


def test_chat_completions_unauthenticated_returns_401(client):
    """POST /v1/chat/completions returns 401 Unauthorized without auth headers."""
    payload = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 401
    assert resp.json().get("detail") == "Unauthorized"


def test_chat_completions_authenticated_standard_non_streaming(client, monkeypatch):
    """POST /v1/chat/completions returns 200 OK with mocked completion for non-streaming call."""
    class Msg:
        role = "assistant"
        content = "Hello from mocked completion!"

    class Choice:
        index = 0
        message = Msg()
        finish_reason = "stop"

    class Usage:
        prompt_tokens = 12
        completion_tokens = 6
        total_tokens = 18

    class MockResp:
        id = "chatcmpl-test-abc123"
        created = 1725642000
        model = "gpt-5-mini"
        choices = [Choice()]
        usage = Usage()

    def _mock_completion(**kwargs):
        return MockResp()

    monkeypatch.setattr(server.academicai, "completion", _mock_completion)

    payload = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
    }
    resp = client.post("/v1/chat/completions", headers=_auth_headers(), json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("id") == "chatcmpl-test-abc123"
    assert data.get("object") == "chat.completion"
    assert data.get("model") == "gpt-5-mini"
    assert isinstance(data.get("choices"), list)
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice.get("index") == 0
    assert choice.get("finish_reason") == "stop"
    assert choice.get("message", {}).get("role") == "assistant"
    assert choice.get("message", {}).get("content") == "Hello from mocked completion!"
    assert data.get("usage") == {
        "prompt_tokens": 12,
        "completion_tokens": 6,
        "total_tokens": 18,
    }
