import json

from fastapi.testclient import TestClient

import server


def test_chat_completion_includes_cost_headers_from_cache(tmp_path):
    cache_path = tmp_path / "cost_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "updated_at": "2026-03-06T08:00:00+00:00",
                "total_cost": 123.45,
                "total_clients": 7,
                "cost_entries": 19,
                "source": "cache",
            }
        ),
        encoding="utf-8",
    )

    old_completion = server.academicai.completion
    old_cache_file = server.COST_CACHE_FILE
    old_cost_enabled = server.ENABLE_COST_MONITORING

    def fake_completion(**kwargs):
        class Msg:
            content = "Hallo aus Test"

        class Choice:
            message = Msg()
            finish_reason = "stop"

        class Usage:
            prompt_tokens = 1
            completion_tokens = 1
            total_tokens = 2

        class Resp:
            id = "test-id"
            created = 123
            model = kwargs.get("model", "gpt-5")
            choices = [Choice()]
            usage = Usage()

        return Resp()

    try:
        server.academicai.completion = fake_completion
        server.COST_CACHE_FILE = str(cache_path)
        server.ENABLE_COST_MONITORING = True

        client = TestClient(server.app)
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {server.API_KEY}"},
            json={
                "model": "gpt-5",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert r.status_code == 200
        assert r.headers.get("X-AcademicAI-Total-Cost") == "123.45"
        assert r.headers.get("X-AcademicAI-Total-Clients") == "7"
        assert r.headers.get("X-AcademicAI-Cost-Entries") == "19"
        assert r.headers.get("X-AcademicAI-Cost-Updated-At") == "2026-03-06T08:00:00+00:00"
        assert r.headers.get("X-AcademicAI-Cost-Stale") in {"true", "false"}

        # Local Request Cost Headers
        assert "X-AcademicAI-Request-Cost" in r.headers
        assert "X-AcademicAI-Input-Cost" in r.headers
        assert "X-AcademicAI-Output-Cost" in r.headers
        assert r.headers.get("X-AcademicAI-Prompt-Tokens") == "1"
        assert r.headers.get("X-AcademicAI-Completion-Tokens") == "1"
        assert r.headers.get("X-AcademicAI-Cost-Currency") == "EUR"
        assert r.headers.get("X-AcademicAI-Cost-Estimated") in {"true", "false"}
    finally:
        server.academicai.completion = old_completion
        server.COST_CACHE_FILE = old_cache_file
        server.ENABLE_COST_MONITORING = old_cost_enabled


def test_chat_completion_streaming_includes_local_cost_headers(monkeypatch):
    old_completion = server.academicai.completion

    def fake_completion(**kwargs):
        class Msg:
            content = "Streaming chunk test"

        class Choice:
            message = Msg()
            finish_reason = "stop"

        class Usage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

        class Resp:
            id = "test-stream-id"
            created = 1234
            model = kwargs.get("model", "gpt-4o")
            choices = [Choice()]
            usage = Usage()

        return Resp()

    try:
        server.academicai.completion = fake_completion
        client = TestClient(server.app)
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {server.API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 200
        assert r.headers.get("X-AcademicAI-Prompt-Tokens") == "10"
        assert r.headers.get("X-AcademicAI-Completion-Tokens") == "5"
        assert "X-AcademicAI-Request-Cost" in r.headers
        assert r.headers.get("X-AcademicAI-Cost-Currency") == "EUR"
    finally:
        server.academicai.completion = old_completion


def test_responses_api_includes_local_cost_headers(monkeypatch):
    old_completion = server.academicai.completion

    def fake_completion(**kwargs):
        class Msg:
            content = "Responses API test"

        class Choice:
            message = Msg()
            finish_reason = "stop"

        class Usage:
            prompt_tokens = 20
            completion_tokens = 10
            total_tokens = 30

        class Resp:
            id = "test-resp-id"
            created = 5678
            model = kwargs.get("model", "gpt-4o")
            choices = [Choice()]
            usage = Usage()

        return Resp()

    try:
        server.academicai.completion = fake_completion
        client = TestClient(server.app)
        r = client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {server.API_KEY}"},
            json={
                "model": "gpt-4o",
                "input": "Write code",
            },
        )
        assert r.status_code == 200
        assert r.headers.get("X-AcademicAI-Prompt-Tokens") == "20"
        assert r.headers.get("X-AcademicAI-Completion-Tokens") == "10"
        assert "X-AcademicAI-Request-Cost" in r.headers
        assert r.headers.get("X-AcademicAI-Cost-Currency") == "EUR"
    finally:
        server.academicai.completion = old_completion


def test_local_cost_disabled_toggle(monkeypatch):
    old_completion = server.academicai.completion
    old_local_toggle = server.ENABLE_LOCAL_COST_TRACKING

    def fake_completion(**kwargs):
        class Msg:
            content = "Toggle test"

        class Choice:
            message = Msg()
            finish_reason = "stop"

        class Usage:
            prompt_tokens = 5
            completion_tokens = 5
            total_tokens = 10

        class Resp:
            id = "test-toggle-id"
            created = 999
            model = kwargs.get("model", "gpt-4o")
            choices = [Choice()]
            usage = Usage()

        return Resp()

    try:
        server.academicai.completion = fake_completion
        server.ENABLE_LOCAL_COST_TRACKING = False

        client = TestClient(server.app)
        r = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {server.API_KEY}"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        assert r.status_code == 200
        assert "X-AcademicAI-Request-Cost" not in r.headers
        assert "X-AcademicAI-Prompt-Tokens" not in r.headers
    finally:
        server.academicai.completion = old_completion
        server.ENABLE_LOCAL_COST_TRACKING = old_local_toggle

