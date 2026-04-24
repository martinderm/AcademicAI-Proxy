import json

import httpx
import pytest

from _e2e import e2e_server


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search memory.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    },
]


def _json_mode_request(e2e_server: dict, user_text: str) -> httpx.Response:
    payload = {
        "model": e2e_server["model"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only a valid JSON object. "
                    "Use either {\"action\":\"tool_call\",...} or {\"action\":\"respond\",...}."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        "response_format": {"type": "json_object"},
        "tools": SAMPLE_TOOLS,
        "tool_choice": "auto",
    }
    return httpx.post(
        f"{e2e_server['base']}/v1/chat/completions",
        headers=e2e_server["headers"],
        json=payload,
        timeout=60,
    )


@pytest.mark.e2e
def test_json_mode_returns_parseable_json(e2e_server):
    response = _json_mode_request(e2e_server, "Was ist 7 mal 8?")
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"].get("content") or ""
    parsed = json.loads(content)
    assert parsed.get("action") in ("respond", "tool_call")


@pytest.mark.e2e
def test_json_mode_web_question_prefers_tool_or_valid_response(e2e_server):
    response = _json_mode_request(e2e_server, "Was ist das Wetter heute in Wien?")
    assert response.status_code == 200
    content = response.json()["choices"][0]["message"].get("content") or ""
    parsed = json.loads(content)
    assert parsed.get("action") in ("tool_call", "respond")
