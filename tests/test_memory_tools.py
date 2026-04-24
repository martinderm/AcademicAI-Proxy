import json

import httpx
import pytest

from _e2e import e2e_server


OPENCLAW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file. Supports text files and images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {"type": "number"},
                    "limit": {"type": "number"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Semantically search memory files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "maxResults": {"type": "number"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_get",
            "description": "Read memory snippets by path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "from": {"type": "number"},
                    "lines": {"type": "number"},
                },
                "required": ["path"],
            },
        },
    },
]


def _post(e2e_server: dict, messages: list) -> httpx.Response:
    return httpx.post(
        f"{e2e_server['base']}/v1/chat/completions",
        headers=e2e_server["headers"],
        json={
            "model": e2e_server["model"],
            "messages": messages,
            "tools": OPENCLAW_TOOLS,
            "tool_choice": "auto",
            "stream": False,
        },
        timeout=60,
    )


@pytest.mark.e2e
def test_memory_search_triggered(e2e_server):
    response = _post(e2e_server, [{"role": "user", "content": "Was haben wir zuletzt uber den AcademicAI Proxy entschieden?"}])
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"].get("tool_calls") or []
    assert calls
    assert calls[0]["function"]["name"] == "memory_search"


@pytest.mark.e2e
def test_memory_get_or_read_triggered_for_file_request(e2e_server):
    response = _post(e2e_server, [{"role": "user", "content": "Lies die Datei MEMORY.md und zeig mir was drin steht."}])
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"].get("tool_calls") or []
    assert calls
    assert calls[0]["function"]["name"] in ("memory_get", "read")


@pytest.mark.e2e
def test_memory_roundtrip_returns_text(e2e_server):
    messages = [
        {"role": "user", "content": "Was weisst du ueber den AcademicAI Proxy aus dem Memory?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_mem001",
                    "type": "function",
                    "function": {"name": "memory_search", "arguments": '{"query": "AcademicAI Proxy"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_mem001",
            "content": json.dumps([
                {"score": 0.95, "path": "MEMORY.md#12", "text": "AcademicAI Proxy: FastAPI auf Port 11435."}
            ], ensure_ascii=False),
        },
    ]

    response = _post(e2e_server, messages)
    assert response.status_code == 200
    body = response.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"].get("content")
    assert not (choice["message"].get("tool_calls") or [])
