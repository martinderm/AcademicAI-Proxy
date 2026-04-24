import httpx
import pytest

from _e2e import e2e_server


@pytest.mark.e2e
def test_openclaw_style_streaming_request_works(e2e_server):
    payload = {
        "model": e2e_server["model"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "store": False,
        "max_completion_tokens": 32000,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "Du bist ein hilfreicher KI-Assistent."},
            {"role": "user", "content": "Sag Hallo auf Deutsch in einem Satz."},
        ],
    }
    response = httpx.post(
        f"{e2e_server['base']}/v1/chat/completions",
        headers=e2e_server["headers"],
        json=payload,
        timeout=60,
    )
    assert response.status_code == 200
    assert "data: [DONE]" in response.text
