import asyncio

from fastapi.testclient import TestClient
import pytest

import academicai
from academicai.humanization import (
    build_humanization_messages,
    _build_humanization_messages,
    is_human_readable_target,
    _is_human_readable_target,
    last_user_text,
    _last_user_text,
    run_humanization_pass,
    _run_humanization_pass,
)
import server


def _run(coro):
    return asyncio.run(coro)


class _MockMsg:
    def __init__(self, content: str):
        self.role = "assistant"
        self.content = content


class _MockChoice:
    def __init__(self, content: str):
        self.index = 0
        self.message = _MockMsg(content)
        self.finish_reason = "stop"


class _MockUsage:
    prompt_tokens = 10
    completion_tokens = 5
    total_tokens = 15


class _MockResp:
    def __init__(self, content: str, model: str = "academicai/gpt-5"):
        self.id = "chatcmpl-humanization-test"
        self.created = 1725642000
        self.model = model
        self.choices = [_MockChoice(content)]
        self.usage = _MockUsage()


# ---------------------------------------------------------------------------
# 1. Module Exports & Aliases
# ---------------------------------------------------------------------------


def test_humanization_module_and_server_exports():
    """Verify all 8 humanization functions/aliases are exported by academicai and server."""
    funcs = [
        "build_humanization_messages",
        "_build_humanization_messages",
        "is_human_readable_target",
        "_is_human_readable_target",
        "last_user_text",
        "_last_user_text",
        "run_humanization_pass",
        "_run_humanization_pass",
    ]
    for fn in funcs:
        assert hasattr(server, fn), f"server is missing {fn}"
        assert hasattr(academicai, fn), f"academicai is missing {fn}"
        assert callable(getattr(server, fn))
        assert callable(getattr(academicai, fn))


# ---------------------------------------------------------------------------
# 2. Target Channel Classification (is_human_readable_target)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channel_marker",
    [
        "channel=whatsapp",
        '"channel": "whatsapp"',
        "channel=telegram",
        '"channel": "telegram"',
        "channel=signal",
        '"channel": "signal"',
        "channel=imessage",
        '"channel": "imessage"',
        "channel=discord",
        '"channel": "discord"',
        "channel=slack",
        '"channel": "slack"',
        "channel=googlechat",
        '"channel": "googlechat"',
        "channel=irc",
        '"channel": "irc"',
        "channel=webchat",
        '"channel": "webchat"',
        '"chat_type": "group"',
        '"chat_type": "direct"',
        "You are a personal assistant running inside OpenClaw.",
    ],
)
def test_is_human_readable_target_system_channel_markers(channel_marker):
    msgs = [
        {"role": "system", "content": f"System context: {channel_marker}"},
        {"role": "user", "content": "How is the weather?"},
    ]
    assert is_human_readable_target(msgs) is True
    assert _is_human_readable_target(msgs) is True
    assert server.is_human_readable_target(msgs) is True


@pytest.mark.parametrize(
    "metadata_marker",
    [
        "Conversation info (untrusted metadata)\nsome metadata here",
        '{"is_group_chat": true}',
        '{"is_group_chat": false}',
        '"conversation_label": "Project Room"',
        '"sender": "+43660123456"',
    ],
)
def test_is_human_readable_target_user_metadata_markers(metadata_marker):
    msgs = [
        {"role": "system", "content": "Generic instructions without channel tag."},
        {"role": "user", "content": f"Payload:\n{metadata_marker}\nActual question: Hello"},
    ]
    assert is_human_readable_target(msgs) is True


def test_is_human_readable_target_machine_cron_overrides_human_markers():
    msgs = [
        {"role": "system", "content": "channel=whatsapp conversation context"},
        {"role": "user", "content": "[cron:hourly_job] Run automated check and return status"},
    ]
    assert is_human_readable_target(msgs) is False
    assert server._is_human_readable_target(msgs) is False


def test_is_human_readable_target_empty_and_default_messages():
    assert is_human_readable_target([]) is False
    assert is_human_readable_target(None) is False
    normal_msgs = [
        {"role": "system", "content": "You are a standard helpful assistant."},
        {"role": "user", "content": "Calculate 2 + 2"},
    ]
    assert is_human_readable_target(normal_msgs) is False

    # Tolerates malformed or non-dict message entries gracefully
    malformed = [None, 123, {"role": "invalid"}, {"other": "content"}]
    assert is_human_readable_target(malformed) is False


# ---------------------------------------------------------------------------
# 3. User Text Extraction (last_user_text)
# ---------------------------------------------------------------------------


def test_last_user_text_simple_string():
    msgs = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Latest question"},
    ]
    assert last_user_text(msgs) == "Latest question"
    assert _last_user_text(msgs) == "Latest question"
    assert server.last_user_text(msgs) == "Latest question"


def test_last_user_text_multipart_list():
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
        }
    ]
    assert last_user_text(msgs) == "Part 1\nPart 2"


def test_last_user_text_empty_and_missing():
    assert last_user_text([]) == ""
    assert last_user_text(None) == ""
    assert last_user_text([{"role": "system", "content": "sys"}]) == ""
    assert last_user_text([{"role": "user", "content": None}]) == ""
    assert last_user_text([{"role": "user", "content": ""}]) == ""
    assert last_user_text([None, 42]) == ""


# ---------------------------------------------------------------------------
# 4. Prompt Construction (build_humanization_messages)
# ---------------------------------------------------------------------------


def test_humanization_prompt_contains_original_question_and_data():
    msgs = server._build_humanization_messages(
        original_user_query="Was weißt du über Zoom-Räume?",
        structured_content='{"rooms":[{"name":"Internal Room"}]}'
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "Return only the final answer text" in msgs[0]["content"]
    assert "Do NOT include JSON, code blocks" in msgs[0]["content"]
    assert "Was weißt du über Zoom-Räume?" in msgs[1]["content"]
    assert '"rooms"' in msgs[1]["content"]


def test_humanization_prompt_empty_and_none_fallbacks():
    msgs_empty = build_humanization_messages("", '{"data": 123}')
    assert len(msgs_empty) == 2
    assert "Original user question:\n-" in msgs_empty[1]["content"]

    msgs_none = _build_humanization_messages(None, None)
    assert len(msgs_none) == 2
    assert "Original user question:\n-" in msgs_none[1]["content"]
    assert "Structured/tool-derived result:\n\n" in msgs_none[1]["content"]


# ---------------------------------------------------------------------------
# 5. Second-Pass Execution (run_humanization_pass)
# ---------------------------------------------------------------------------


def test_humanization_pass_success():
    def fake_sync_completion(**kwargs):
        return _MockResp("Natürliche Antwort ohne Meta.")

    old_completion = server.academicai.completion
    old_model = server.HUMANIZATION_MODEL
    old_temp = server.HUMANIZATION_TEMPERATURE
    try:
        server.academicai.completion = fake_sync_completion
        server.HUMANIZATION_MODEL = ""
        server.HUMANIZATION_TEMPERATURE = 0.2

        out = _run(server._run_humanization_pass(
            model="academicai/gpt-5",
            original_user_query="Welche Zoom Räume kennst du?",
            structured_content='{"status":"ok","rooms":[{"name":"A","url":"https://x"}]}'
        ))

        assert out == "Natürliche Antwort ohne Meta."
    finally:
        server.academicai.completion = old_completion
        server.HUMANIZATION_MODEL = old_model
        server.HUMANIZATION_TEMPERATURE = old_temp


def test_humanization_pass_fallback_on_error():
    def boom(**kwargs):
        raise RuntimeError("backend down")

    old_completion = server.academicai.completion
    try:
        server.academicai.completion = boom
        out = _run(server._run_humanization_pass(
            model="academicai/gpt-5",
            original_user_query="Was weißt du?",
            structured_content='{"status":"ok"}'
        ))
        assert out is None
    finally:
        server.academicai.completion = old_completion


def test_humanization_pass_fallback_on_empty_content():
    def empty_completion(**kwargs):
        return _MockResp("   \n  ")

    old_completion = server.academicai.completion
    try:
        server.academicai.completion = empty_completion
        out = _run(run_humanization_pass(
            model="academicai/gpt-5",
            original_user_query="Hello",
            structured_content="some content",
        ))
        assert out is None
    finally:
        server.academicai.completion = old_completion


def test_humanization_pass_async_callable_support():
    async def async_fake_completion(**kwargs):
        return _MockResp("Async humanized response.")

    old_completion = server.academicai.completion
    try:
        server.academicai.completion = async_fake_completion
        out = _run(run_humanization_pass(
            model="academicai/gpt-5",
            original_user_query="Hello async",
            structured_content="data",
        ))
        assert out == "Async humanized response."
    finally:
        server.academicai.completion = old_completion


# ---------------------------------------------------------------------------
# 6. Formatting & Server Chat Endpoint Pass Integration
# ---------------------------------------------------------------------------


def test_human_text_formatter_removes_common_meta_keys():
    from academicai.tool_emulation import format_arbitrary_json_for_humans

    src = '''{
      "status": "ok",
      "source": "memory/references/ZOOM-ROOMS.md",
      "timestamp": "2026-02-27T17:00:00Z",
      "rooms": [{"name":"A","url":"https://x"}]
    }'''
    out = format_arbitrary_json_for_humans(src)
    assert out is not None
    assert "status" not in out.lower()
    assert "source" not in out.lower()
    assert "timestamp" not in out.lower()
    assert "A: https://x" in out


def test_chat_endpoint_respects_enable_humanization_pass_disabled(monkeypatch):
    """When ENABLE_HUMANIZATION_PASS is False, humanization pass must not run."""
    client = TestClient(server.app)
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs)
        # First-pass response with respond action
        return _MockResp('{"action": "respond", "content": "Raw first-pass response"}')

    monkeypatch.setattr(server.academicai, "completion", mock_completion)
    monkeypatch.setattr(server, "ENABLE_HUMANIZATION_PASS", False)

    payload = {
        "model": "academicai/gpt-5",
        "messages": [
            {"role": "system", "content": "channel=whatsapp"},
            {"role": "user", "content": "Status update?"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_status", "parameters": {}}}],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {server.API_KEY}"}
    resp = client.post("/v1/chat/completions", headers=headers, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Raw first-pass response"
    # Exactly one completion call was made (no second pass)
    assert len(calls) == 1


def test_chat_endpoint_respects_enable_humanization_pass_enabled(monkeypatch):
    """When ENABLE_HUMANIZATION_PASS is True and human channel detected, second pass runs."""
    client = TestClient(server.app)
    calls = []

    def mock_completion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # First pass: tool emulation response
            return _MockResp('{"action": "respond", "content": "Raw first-pass response"}')
        # Second pass: humanization rewrite
        return _MockResp("Final polished human response.")

    monkeypatch.setattr(server.academicai, "completion", mock_completion)
    monkeypatch.setattr(server, "ENABLE_HUMANIZATION_PASS", True)

    payload = {
        "model": "academicai/gpt-5",
        "messages": [
            {"role": "system", "content": "channel=whatsapp"},
            {"role": "user", "content": "Status update?"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_status", "parameters": {}}}],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {server.API_KEY}"}
    resp = client.post("/v1/chat/completions", headers=headers, json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Final polished human response."
    # Two completion calls were made: first pass + second humanization pass
    assert len(calls) == 2
    assert "Original user question" in calls[1]["messages"][1]["content"]
