import json

import httpx
import pytest

from _e2e import e2e_server
from _local_env import BASE, auth_headers
from academicai.tool_emulation import (
    extract_respond_content,
    format_arbitrary_json_as_codeblock,
    format_arbitrary_json_for_humans,
    parse_json_mode_response,
)


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
    }
]


def _post(payload: dict) -> httpx.Response:
    return httpx.post(
        f"{BASE}/v1/chat/completions",
        headers=auth_headers(),
        json=payload,
        timeout=60,
    )


@pytest.mark.e2e
def test_no_tools_still_returns_normal_response(e2e_server):
    response = _post(
        {
            "model": e2e_server["model"],
            "messages": [{"role": "user", "content": "Antworte mit genau: HALLO_TEST"}],
            "stream": False,
        }
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("content")


@pytest.mark.e2e
def test_tools_can_emit_tool_calls(e2e_server):
    response = _post(
        {
            "model": e2e_server["model"],
            "messages": [{"role": "user", "content": "Suche im Web nach OpenClaw. Nutze das web_search Tool."}],
            "tools": SAMPLE_TOOLS,
            "tool_choice": "auto",
            "stream": False,
        }
    )
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    if choice["finish_reason"] == "tool_calls":
        calls = choice["message"].get("tool_calls") or []
        assert calls
        assert calls[0]["function"]["name"] == "web_search"
    else:
        assert choice["finish_reason"] == "stop"
        assert choice["message"].get("content")


def test_parse_json_mode_response_handles_wrapped_json():
    payload = "Hier der Call:\n```json\n" + json.dumps({"action": "tool_call", "name": "exec", "arguments": {"command": "Get-Date"}}) + "\n```"
    parsed = parse_json_mode_response(payload)
    assert parsed is not None
    assert parsed["action"] == "tool_call"
    assert parsed["name"] == "exec"


def test_extract_respond_content_and_fallback_formatters():
    respond = json.dumps({"action": "respond", "content": "HEARTBEAT_OK"})
    assert extract_respond_content(respond) == "HEARTBEAT_OK"

    arbitrary = json.dumps({"job": "wp-sync", "steps": ["list", "parse"]})
    codeblock = format_arbitrary_json_as_codeblock(arbitrary)
    assert codeblock is not None
    assert codeblock.startswith("```json")

    fenced = "```json\n{\"status\": \"ok\", \"capabilities\": [\"zoom_rooms\"]}\n```"
    human = format_arbitrary_json_for_humans(fenced)
    assert isinstance(human, str)
    assert human.startswith("Hier die Infos:")
