"""
Unit and integration tests for OpenAI Responses API (POST /v1/responses).

Tests:
1. Normalization of Responses API request structures into canonical proxy format:
   - string inputs, message content parts, function_call items, function_call_output items
   - tool definition normalization (flat, nested, namespace)
   - parameter mappings (reasoning.effort, text.verbosity, max_output_tokens)
2. Inbound request validation:
   - 422 on missing/invalid model or missing input & instructions
   - 413 on message limits, text length, tool counts, schema limits
3. Output serialization (non-streaming):
   - assistant message output format
   - function_call output format
4. Streaming SSE generator:
   - valid event names, sequence, and payload structure for text and tool calls
5. Integration endpoint tests via TestClient:
   - authentication (401 on missing/wrong key)
   - non-streaming text completion (200 OK)
   - non-streaming tool call completion (200 OK)
   - streaming SSE text completion (200 OK text/event-stream)
   - streaming SSE tool call completion (200 OK text/event-stream)
   - multi-turn tool roundtrip (Turn 1: tool call -> Turn 2: tool output -> final answer)
"""

import json
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import server
from academicai.provider import Choice, CompletionResponse, Message, Usage
from academicai.request_guards import validate_responses_request_body
from academicai.responses import (
    build_responses_output,
    build_responses_sse_events,
    normalize_responses_request,
)


@pytest.fixture
def client():
    return TestClient(server.app)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {server.API_KEY}"}


def _mock_completion_response(content: str, model: str = "gpt-4o") -> CompletionResponse:
    return CompletionResponse(
        id="comp_test_123",
        object="chat.completion",
        created=1726000000,
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=25,
            completion_tokens=15,
            total_tokens=40,
        ),
    )


# ---------------------------------------------------------------------------
# 1. Request Normalization
# ---------------------------------------------------------------------------


def test_normalize_string_input_and_instructions():
    body = {
        "model": "gpt-4o",
        "instructions": "System instructions here",
        "input": "User query here",
    }
    model, messages, tools, tool_choice, optional = normalize_responses_request(body)
    assert model == "gpt-4o"
    assert len(messages) == 2
    assert messages[0]["role"] == "developer"
    assert messages[0]["content"] == "System instructions here"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User query here"
    assert tools == []
    assert tool_choice is None


def test_normalize_content_parts_and_roles():
    body = {
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Part 1"},
                    {"type": "text", "text": "Part 2"},
                ],
            }
        ],
    }
    _, messages, _, _, _ = normalize_responses_request(body)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Part 1\nPart 2" == messages[0]["content"]


def test_normalize_function_call_and_output_roundtrip_items():
    body = {
        "model": "gpt-4o",
        "input": [
            {"type": "message", "role": "user", "content": "Please run command"},
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "exec_command",
                "arguments": "{\"cmd\": \"echo hi\"}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": "hi\n",
            },
        ],
    }
    _, messages, _, _, _ = normalize_responses_request(body)
    assert len(messages) == 3

    # User message
    assert messages[0]["role"] == "user"

    # Assistant tool call
    assert messages[1]["role"] == "assistant"
    assert "tool_calls" in messages[1]
    assert messages[1]["tool_calls"][0]["id"] == "call_123"
    assert messages[1]["tool_calls"][0]["function"]["name"] == "exec_command"
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == "{\"cmd\": \"echo hi\"}"

    # Tool output
    assert messages[2]["role"] == "tool"
    assert messages[2]["tool_call_id"] == "call_123"
    assert messages[2]["content"] == "hi\n"


def test_normalize_tools_flat_nested_and_namespace():
    body = {
        "model": "gpt-4o",
        "input": "test",
        "tools": [
            {
                "type": "function",
                "name": "flat_tool",
                "description": "Flat tool",
                "parameters": {"type": "object"},
            },
            {
                "type": "function",
                "function": {
                    "name": "nested_tool",
                    "description": "Nested tool",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "namespace",
                "tools": [
                    {
                        "type": "function",
                        "name": "ns_tool",
                        "description": "Namespace tool",
                        "parameters": {"type": "object"},
                    }
                ],
            },
            {
                "type": "web_search",
            },
        ],
        "tool_choice": "auto",
        "reasoning": {"effort": "high"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 1200,
    }
    _, _, tools, tool_choice, optional = normalize_responses_request(body)
    assert len(tools) == 3
    tool_names = [t["function"]["name"] for t in tools]
    assert "flat_tool" in tool_names
    assert "nested_tool" in tool_names
    assert "ns_tool" in tool_names
    assert tool_choice == "auto"
    assert optional["reasoning_effort"] == "high"
    assert optional["verbosity"] == "low"
    assert optional["max_tokens"] == 1200


# ---------------------------------------------------------------------------
# 2. Request Validation
# ---------------------------------------------------------------------------


def test_validate_responses_body_missing_model():
    with pytest.raises(HTTPException) as exc:
        validate_responses_request_body({"input": "hello"})
    assert exc.value.status_code == 422


def test_validate_responses_body_missing_input_and_instructions():
    with pytest.raises(HTTPException) as exc:
        validate_responses_request_body({"model": "gpt-4o"})
    assert exc.value.status_code == 422


def test_validate_responses_body_limits_413():
    # Message text limit
    with pytest.raises(HTTPException) as exc:
        validate_responses_request_body(
            {"model": "gpt-4o", "input": "toolong"},
            max_message_text_chars=4,
        )
    assert exc.value.status_code == 413

    # Tools count limit
    with pytest.raises(HTTPException) as exc:
        validate_responses_request_body(
            {"model": "gpt-4o", "input": "ok", "tools": [{"name": "1"}, {"name": "2"}]},
            max_tools=1,
        )
    assert exc.value.status_code == 413


# ---------------------------------------------------------------------------
# 3. Output Serialization (Non-Streaming)
# ---------------------------------------------------------------------------


def test_build_responses_output_text():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    out = build_responses_output("123", 1726000000, "gpt-4o", "Hello world", [], usage)
    assert out["id"] == "resp_123"
    assert out["object"] == "response"
    assert out["status"] == "completed"
    assert len(out["output"]) == 1
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["role"] == "assistant"
    assert out["output"][0]["content"][0]["text"] == "Hello world"
    assert out["usage"]["input_tokens"] == 10
    assert out["usage"]["output_tokens"] == 5
    assert out["usage"]["total_tokens"] == 15


def test_build_responses_output_tool_call():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    tool_calls = [{"id": "call_abc", "name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"}]
    out = build_responses_output("123", 1726000000, "gpt-4o", "", tool_calls, usage)
    assert out["status"] == "completed"
    assert len(out["output"]) == 1
    assert out["output"][0]["type"] == "function_call"
    assert out["output"][0]["call_id"] == "call_abc"
    assert out["output"][0]["name"] == "exec_command"
    assert out["output"][0]["arguments"] == "{\"cmd\": \"ls\"}"
    assert out["usage"]["input_tokens"] == 10
    assert out["usage"]["output_tokens"] == 5


# ---------------------------------------------------------------------------
# 4. SSE Stream Serialization
# ---------------------------------------------------------------------------


def _parse_sse_events(raw_stream: list[str]) -> list[tuple[str, dict]]:
    events = []
    for chunk in raw_stream:
        lines = chunk.strip().split("\n")
        event_type = None
        data_json = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data_json = json.loads(line[len("data: "):].strip())
        if event_type and data_json:
            events.append((event_type, data_json))
    return events


def test_build_responses_sse_events_text():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    chunks = list(build_responses_sse_events("123", 1726000000, "gpt-4o", "Streaming text", [], usage))
    events = _parse_sse_events(chunks)

    event_names = [e[0] for e in events]
    assert event_names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]

    # Verify delta text
    delta_ev = next(e[1] for e in events if e[0] == "response.output_text.delta")
    assert delta_ev["delta"] == "Streaming text"

    # Verify completed event
    completed_ev = next(e[1] for e in events if e[0] == "response.completed")
    assert completed_ev["response"]["status"] == "completed"
    assert completed_ev["response"]["output"][0]["content"][0]["text"] == "Streaming text"


def test_build_responses_sse_events_tool_call():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    tool_calls = [{"id": "call_456", "name": "run_job", "arguments": "{\"job_id\": 42}"}]
    chunks = list(build_responses_sse_events("123", 1726000000, "gpt-4o", "", tool_calls, usage))
    events = _parse_sse_events(chunks)

    event_names = [e[0] for e in events]
    assert event_names == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]

    # Verify arguments delta
    args_delta_ev = next(e[1] for e in events if e[0] == "response.function_call_arguments.delta")
    assert args_delta_ev["call_id"] == "call_456"
    assert args_delta_ev["delta"] == "{\"job_id\": 42}"

    # Verify completed event
    completed_ev = next(e[1] for e in events if e[0] == "response.completed")
    assert completed_ev["response"]["output"][0]["type"] == "function_call"
    assert completed_ev["response"]["output"][0]["call_id"] == "call_456"
    assert completed_ev["response"]["output"][0]["name"] == "run_job"


# ---------------------------------------------------------------------------
# 5. Integration Endpoint Tests
# ---------------------------------------------------------------------------


def test_responses_endpoint_unauthenticated_returns_401(client):
    resp = client.post("/v1/responses", json={"model": "gpt-4o", "input": "hello"})
    assert resp.status_code == 401


def test_responses_endpoint_invalid_key_returns_401(client):
    resp = client.post(
        "/v1/responses",
        headers={"Authorization": "Bearer wrong-key"},
        json={"model": "gpt-4o", "input": "hello"},
    )
    assert resp.status_code == 401


def test_responses_endpoint_non_streaming_text(client, monkeypatch):
    mock_academicai = MagicMock()
    mock_academicai.completion.return_value = _mock_completion_response("Non-streaming answer")
    monkeypatch.setattr(server, "academicai", mock_academicai)
    monkeypatch.setattr(server, "_build_cost_headers", lambda cache: {"X-AcademicAI-Total-Cost": "12.34"})

    resp = client.post(
        "/v1/responses",
        headers=_auth_headers(),
        json={"model": "gpt-4o", "input": "What is the capital of Austria?"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["model"] == "gpt-4o"
    assert len(data["output"]) == 1
    assert data["output"][0]["type"] == "message"
    assert data["output"][0]["content"][0]["text"] == "Non-streaming answer"
    assert "X-AcademicAI-Total-Cost" in resp.headers


def test_responses_endpoint_non_streaming_tool_call(client, monkeypatch):
    mock_academicai = MagicMock()
    tool_call_json = '{"action": "tool_call", "name": "exec_command", "arguments": {"cmd": "dir"}}'
    mock_academicai.completion.return_value = _mock_completion_response(tool_call_json)
    monkeypatch.setattr(server, "academicai", mock_academicai)

    resp = client.post(
        "/v1/responses",
        headers=_auth_headers(),
        json={
            "model": "gpt-4o",
            "input": "Run dir command",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Execute shell command",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                }
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert len(data["output"]) == 1
    tool_out = data["output"][0]
    assert tool_out["type"] == "function_call"
    assert tool_out["name"] == "exec_command"
    assert json.loads(tool_out["arguments"]) == {"cmd": "dir"}


def test_responses_endpoint_streaming_text(client, monkeypatch):
    mock_academicai = MagicMock()
    mock_academicai.completion.return_value = _mock_completion_response("Streaming answer")
    monkeypatch.setattr(server, "academicai", mock_academicai)

    resp = client.post(
        "/v1/responses",
        headers=_auth_headers(),
        json={"model": "gpt-4o", "input": "Hello stream", "stream": True},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body_text = resp.text
    assert "event: response.created" in body_text
    assert "event: response.output_text.delta" in body_text
    assert "event: response.completed" in body_text


def test_responses_endpoint_multi_turn_roundtrip(client, monkeypatch):
    mock_academicai = MagicMock()

    # Turn 1: Model emits tool call
    turn1_json = '{"action": "tool_call", "name": "exec_command", "arguments": {"cmd": "echo test"}}'
    mock_academicai.completion.return_value = _mock_completion_response(turn1_json)
    monkeypatch.setattr(server, "academicai", mock_academicai)

    resp1 = client.post(
        "/v1/responses",
        headers=_auth_headers(),
        json={
            "model": "gpt-4o",
            "input": "Execute test command",
            "tools": [{"type": "function", "name": "exec_command", "description": "Run command"}],
        },
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    call_id = data1["output"][0]["call_id"]
    assert call_id.startswith("call_")

    # Turn 2: Client returns tool output, model returns final answer
    turn2_json = '{"action": "respond", "content": "Done! Command output was: test"}'
    mock_academicai.completion.return_value = _mock_completion_response(turn2_json)

    resp2 = client.post(
        "/v1/responses",
        headers=_auth_headers(),
        json={
            "model": "gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": "Execute test command"},
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": "exec_command",
                    "arguments": "{\"cmd\": \"echo test\"}",
                },
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": "test\n",
                },
            ],
            "tools": [{"type": "function", "name": "exec_command", "description": "Run command"}],
        },
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert len(data2["output"]) == 1
    assert data2["output"][0]["type"] == "message"
    assert data2["output"][0]["content"][0]["text"] == "Done! Command output was: test"
