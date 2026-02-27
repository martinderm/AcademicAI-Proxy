import server


def test_post_tool_guard_added_when_last_message_is_tool_and_tools_enabled():
    messages = [
        {"role": "user", "content": "Suche X"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"X"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Result: ..."},
    ]

    out = server._apply_post_tool_guard(messages, has_tools=True)

    assert len(out) == len(messages) + 1
    assert out[0]["role"] == "system"
    assert "NO_FURTHER_TOOL_CALLS" in out[0]["content"]


def test_post_tool_guard_not_added_without_tool_mode():
    messages = [
        {"role": "user", "content": "Hallo"},
        {"role": "assistant", "content": "Hi"},
    ]

    out = server._apply_post_tool_guard(messages, has_tools=False)

    assert out == messages


def test_enforce_write_before_mail_delete_blocks_delete_without_write():
    calls = [
        {"name": "exec", "arguments": {"command": "& $wrapper message delete 257"}},
        {"name": "exec", "arguments": {"command": "Get-Date"}},
    ]

    out, blocked = server._enforce_write_before_mail_delete(calls)

    # delete blockiert, normales exec bleibt
    assert blocked is True
    assert len(out) == 1
    assert out[0]["name"] == "exec"
    assert "Get-Date" in out[0]["arguments"]["command"]


def test_enforce_write_before_mail_delete_allows_delete_after_write():
    calls = [
        {"name": "write", "arguments": {"path": "WEB-ADMIN.md", "content": "ok"}},
        {"name": "exec", "arguments": {"command": "& $wrapper message delete 257"}},
    ]

    out, blocked = server._enforce_write_before_mail_delete(calls)

    assert blocked is False
    assert len(out) == 2
    assert out[0]["name"] == "write"
    assert out[1]["name"] == "exec"
