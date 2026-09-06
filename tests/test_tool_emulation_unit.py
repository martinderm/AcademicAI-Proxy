import json
import pytest

from academicai.tool_emulation import (
    _compact_tool_def,
    _format_param_type,
    _format_default_val,
    _repair_and_load_json,
    _extract_json_object,
    _normalize_tool_call_entry,
    build_json_mode_system_prompt,
    inject_tools_into_messages,
    parse_tool_calls,
    parse_json_mode_response,
)
from academicai.transformation import _normalize_messages


# ---------------------------------------------------------------------------
# 1. TypeScript-style signature rendering tests
# ---------------------------------------------------------------------------

def test_compact_tool_def_enums_and_truncation():
    tool_short_enum = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["exact", "regex", "fuzzy"],
                    }
                },
            },
        },
    }
    sig = _compact_tool_def(tool_short_enum)
    assert sig == '- search(mode?: "exact" | "regex" | "fuzzy") -- Search items.'

    tool_long_enum = {
        "type": "function",
        "function": {
            "name": "filter",
            "description": "Filter by status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["a", "b", "c", "d", "e", "f"],
                    }
                },
            },
        },
    }
    sig_long = _compact_tool_def(tool_long_enum)
    assert sig_long == '- filter(status?: "a" | "b" | "c" | "d" | ...) -- Filter by status.'


def test_compact_tool_def_typed_arrays():
    tool = {
        "type": "function",
        "function": {
            "name": "batch_process",
            "description": "Process multiple entries.",
            "parameters": {
                "type": "object",
                "required": ["ids"],
                "properties": {
                    "ids": {"type": "array", "items": {"type": "integer"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "flags": {"type": "array", "items": {"type": "boolean"}},
                    "metadata": {"type": "array", "items": {"type": "object"}},
                    "raw": {"type": "array"},
                },
            },
        },
    }
    sig = _compact_tool_def(tool)
    assert "ids: number[]" in sig
    assert "tags?: string[]" in sig
    assert "flags?: boolean[]" in sig
    assert "metadata?: object[]" in sig
    assert "raw?: any[]" in sig
    assert sig.startswith("- batch_process(")
    assert sig.endswith(") -- Process multiple entries.")


def test_compact_tool_def_defaults():
    tool = {
        "type": "function",
        "function": {
            "name": "paginate",
            "description": "Paginate records.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10},
                    "mode": {"enum": ["exact", "regex"], "default": "exact"},
                    "verbose": {"type": "boolean", "default": False},
                    "factor": {"type": "number", "default": 1.5},
                },
            },
        },
    }
    sig = _compact_tool_def(tool)
    assert "limit?: number = 10" in sig
    assert 'mode?: "exact" | "regex" = "exact"' in sig
    assert "verbose?: boolean = false" in sig
    assert "factor?: number = 1.5" in sig


def test_compact_tool_def_shallow_objects_and_optionality():
    tool = {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": "Query database.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "filter": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "tags": {"type": "array"},
                        },
                    },
                    "generic_cfg": {"type": "object"},
                },
            },
        },
    }
    sig = _compact_tool_def(tool)
    assert "query: string" in sig  # required, no '?'
    assert "filter?: {query, tags}" in sig  # optional shallow object
    assert "generic_cfg?: object" in sig  # optional generic object
    assert sig == "- query_db(query: string, filter?: {query, tags}, generic_cfg?: object) -- Query database."


# ---------------------------------------------------------------------------
# 2. Hard enforcement of tool_choice
# ---------------------------------------------------------------------------

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
            "name": "lookup_user",
            "description": "Look up user info.",
            "parameters": {
                "type": "object",
                "required": ["user_id"],
                "properties": {"user_id": {"type": "string"}},
            },
        },
    },
]


def test_tool_choice_required_enforcement():
    prompt = build_json_mode_system_prompt(SAMPLE_TOOLS, tool_choice="required")
    assert "MANDATORY" in prompt
    assert '{"action": "respond"}' in prompt
    assert "FORBIDDEN" in prompt

    msgs = [{"role": "user", "content": "Tell me a joke"}]
    injected = inject_tools_into_messages(msgs, SAMPLE_TOOLS, tool_choice="required")
    assert injected[0]["role"] == "system"
    assert "MANDATORY" in injected[0]["content"]
    assert injected[-1]["role"] == "user"
    assert "json" in injected[-1]["content"].lower()
    assert '{"action": "respond"}' in injected[-1]["content"]


def test_tool_choice_specific_tool_dict_enforcement():
    choice = {"type": "function", "function": {"name": "lookup_user"}}
    prompt = build_json_mode_system_prompt(SAMPLE_TOOLS, tool_choice=choice)
    assert "lookup_user" in prompt
    assert "MANDATORY" in prompt or "SPECIFIC TOOL" in prompt
    assert "FORBIDDEN" in prompt
    assert '{"action": "respond"}' in prompt

    msgs = [{"role": "user", "content": "Find user 42"}]
    injected = inject_tools_into_messages(msgs, SAMPLE_TOOLS, tool_choice=choice)
    assert "lookup_user" in injected[-1]["content"]


def test_tool_choice_specific_tool_str_enforcement():
    prompt = build_json_mode_system_prompt(SAMPLE_TOOLS, tool_choice="lookup_user")
    assert "lookup_user" in prompt
    assert "FORBIDDEN" in prompt

    msgs = [{"role": "user", "content": "Find user 42"}]
    injected = inject_tools_into_messages(msgs, SAMPLE_TOOLS, tool_choice="lookup_user")
    assert "lookup_user" in injected[-1]["content"]


def test_tool_choice_auto_and_none():
    prompt_auto = build_json_mode_system_prompt(SAMPLE_TOOLS, tool_choice="auto")
    assert "MANDATORY" not in prompt_auto

    prompt_none = build_json_mode_system_prompt(SAMPLE_TOOLS, tool_choice="none")
    assert "MUST NOT call any tools" in prompt_none


# ---------------------------------------------------------------------------
# 3. JSON-repair sanitization
# ---------------------------------------------------------------------------

def test_json_repair_trailing_commas():
    # Trailing comma in dict
    raw_dict = '{"action": "tool_call", "name": "exec", "arguments": {"cmd": "dir",},}'
    parsed = _extract_json_object(raw_dict)
    assert parsed is not None
    assert parsed["action"] == "tool_call"
    assert parsed["name"] == "exec"
    assert parsed["arguments"] == {"cmd": "dir"}

    # Trailing comma in array
    raw_array = '{"action": "tool_calls", "calls": [{"name": "foo", "arguments": {}},],}'
    calls = parse_tool_calls(raw_array)
    assert len(calls) == 1
    assert calls[0]["name"] == "foo"


def test_json_repair_unescaped_control_characters():
    # Unescaped newline and tab inside JSON string
    raw = '{"action": "respond", "content": "Line 1\nLine 2\tTabbed"}'
    parsed = _extract_json_object(raw)
    assert parsed is not None
    assert parsed["action"] == "respond"
    assert "Line 1\nLine 2\tTabbed" in parsed["content"]


def test_json_repair_borderline_arguments_string():
    entry = {
        "name": "write_file",
        "arguments": '{"path": "C:\\temp\\file.txt", "content": "data",}',
    }
    normalized = _normalize_tool_call_entry(entry)
    assert normalized is not None
    assert normalized["name"] == "write_file"
    assert normalized["arguments"]["content"] == "data"

    # Borderline unparseable string falls back to empty dict without throwing
    broken_entry = {"name": "broken", "arguments": "NOT_JSON"}
    normalized_broken = _normalize_tool_call_entry(broken_entry)
    assert normalized_broken == {"name": "broken", "arguments": {}}


def test_json_repair_in_code_fence():
    text = (
        "Here is the result:\n"
        "```json\n"
        '{\n  "action": "tool_call",\n  "name": "calc",\n  "arguments": {"x": 1, "y": 2,},\n}\n'
        "```\n"
        "Done."
    )
    parsed = _extract_json_object(text)
    assert parsed is not None
    assert parsed["name"] == "calc"
    assert parsed["arguments"] == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# 4. Standardized observation tags (<tool_result>)
# ---------------------------------------------------------------------------

def test_tool_result_tag_with_id_and_name():
    messages = [
        {"role": "user", "content": "Search for Python"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "web_search", "arguments": '{"q": "Python"}'}}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_12345",
            "name": "web_search",
            "content": "Python is a programming language.",
        },
    ]
    normalized = _normalize_messages(messages)
    tool_user_turn = [m for m in normalized if m["role"] == "user"][-1]
    assert '<tool_result id="call_12345" name="web_search">' in tool_user_turn["content"]
    assert "Python is a programming language." in tool_user_turn["content"]
    assert "</tool_result>" in tool_user_turn["content"]


def test_tool_result_tag_with_id_only():
    messages = [
        {"role": "user", "content": "Run tool"},
        {"role": "tool", "tool_call_id": "call_abc", "content": "Result OK"},
    ]
    normalized = _normalize_messages(messages)
    tool_user_turn = [m for m in normalized if m["role"] == "user"][-1]
    assert '<tool_result id="call_abc">' in tool_user_turn["content"]
    assert 'name="' not in tool_user_turn["content"]
    assert "</tool_result>" in tool_user_turn["content"]


def test_tool_result_tag_with_name_only():
    messages = [
        {"role": "user", "content": "Run tool"},
        {"role": "tool", "name": "bash", "content": "echo hi"},
    ]
    normalized = _normalize_messages(messages)
    tool_user_turn = [m for m in normalized if m["role"] == "user"][-1]
    assert '<tool_result name="bash">' in tool_user_turn["content"]
    assert 'id="' not in tool_user_turn["content"]
    assert "</tool_result>" in tool_user_turn["content"]


def test_tool_result_tag_without_attributes():
    messages = [
        {"role": "user", "content": "Run tool"},
        {"role": "tool", "content": "bare result"},
    ]
    normalized = _normalize_messages(messages)
    tool_user_turn = [m for m in normalized if m["role"] == "user"][-1]
    assert "<tool_result>\nbare result\n</tool_result>" in tool_user_turn["content"]
