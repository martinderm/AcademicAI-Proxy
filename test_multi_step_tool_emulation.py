import json

from academicai.tool_emulation import (
    parse_tool_calls,
    build_tool_calls_response,
)


def test_parse_tool_calls_single_legacy_shape():
    content = json.dumps({
        "action": "tool_call",
        "name": "write",
        "arguments": {"path": "WEB-ADMIN.md", "content": "ok"},
    })

    calls = parse_tool_calls(content)

    assert len(calls) == 1
    assert calls[0]["name"] == "write"
    assert calls[0]["arguments"]["path"] == "WEB-ADMIN.md"


def test_parse_tool_calls_new_multi_shape():
    content = json.dumps({
        "action": "tool_calls",
        "calls": [
            {"name": "write", "arguments": {"path": "WEB-ADMIN.md", "content": "ok"}},
            {"name": "exec", "arguments": {"command": "& $wrapper message delete 257"}},
        ],
    })

    calls = parse_tool_calls(content)

    assert len(calls) == 2
    assert [c["name"] for c in calls] == ["write", "exec"]


def test_parse_tool_calls_ignores_invalid_entries():
    content = json.dumps({
        "action": "tool_calls",
        "calls": [
            {"name": "write", "arguments": {"path": "A", "content": "B"}},
            {"name": "", "arguments": {}},
            {"foo": "bar"},
        ],
    })

    calls = parse_tool_calls(content)

    assert len(calls) == 1
    assert calls[0]["name"] == "write"


def test_build_tool_calls_response_supports_multiple_calls():
    response = build_tool_calls_response(
        completion_id="cmpl_test",
        created_ts=1,
        model="gpt-5-mini",
        tool_call_data=[
            {"name": "write", "arguments": {"path": "WEB-ADMIN.md", "content": "ok"}},
            {"name": "exec", "arguments": {"command": "& $wrapper message delete 257"}},
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    tool_calls = response["choices"][0]["message"]["tool_calls"]
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "write"
    assert tool_calls[1]["function"]["name"] == "exec"
