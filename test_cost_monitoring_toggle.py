from fastapi.testclient import TestClient

import server


def test_cost_headers_absent_when_cost_monitoring_disabled():
    old_completion = server.academicai.completion
    old_enabled = server.ENABLE_COST_MONITORING

    def fake_completion(**kwargs):
        class Msg:
            content = "Hallo"

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
        server.ENABLE_COST_MONITORING = False

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
        assert r.headers.get("X-AcademicAI-Total-Cost") is None
        assert r.headers.get("X-AcademicAI-Cost-Updated-At") is None
        assert r.headers.get("X-AcademicAI-Cost-Stale") is None
    finally:
        server.academicai.completion = old_completion
        server.ENABLE_COST_MONITORING = old_enabled
