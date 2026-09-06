"""
Unit tests for academicai.request_guards module.

Validates:
- Success paths for valid chat completion request payloads
- 422 Unprocessable Entity error cases for malformed inputs
- 413 Payload Too Large error cases for threshold limits (messages, chars, tools, schema, JSON body)
- 429 Too Many Requests rate-limiting enforcement
- Bucket sweep / TTL cleanup mechanism purging expired buckets and timestamps
- Server backward compatibility re-exports and monkeypatch compatibility
"""

import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import academicai.request_guards as rg
import server


def _dummy_request(host: str = "127.0.0.1"):
    return SimpleNamespace(client=SimpleNamespace(host=host))


# --- 1. Text extraction tests ---


def test_extract_text_content_string():
    assert rg.extract_text_content("hello world") == "hello world"


def test_extract_text_content_list_of_parts():
    parts = [
        {"type": "text", "text": "line 1"},
        {"type": "image_url", "image_url": {"url": "http://example.com"}},
        {"type": "text", "text": "line 2"},
    ]
    assert rg.extract_text_content(parts) == "line 1\nline 2"


def test_extract_text_content_invalid_or_empty():
    assert rg.extract_text_content(None) == ""
    assert rg.extract_text_content(12345) == ""
    assert rg.extract_text_content([]) == ""


# --- 2. Success path tests ---


def test_validate_chat_request_body_clean_success():
    body = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    # Must not raise
    rg.validate_chat_request_body(body)


def test_validate_chat_request_body_with_tools_success():
    body = {
        "model": "gpt-5-mini",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": [{"type": "text", "text": "Check mail"}]},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "check_mail",
                    "description": "Checks inbox",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    rg.validate_chat_request_body(body)


def test_validate_chat_request_body_with_functions_alias_success():
    body = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "functions": [{"name": "lookup", "parameters": {}}],
    }
    rg.validate_chat_request_body(body)


# --- 3. 422 Error cases ---


def test_validate_chat_request_body_rejects_non_dict():
    for invalid_body in ["string", 123, None, [1, 2, 3]]:
        with pytest.raises(HTTPException) as exc_info:
            rg.validate_chat_request_body(invalid_body)
        assert exc_info.value.status_code == 422
        assert "request body must be a JSON object" in exc_info.value.detail


def test_validate_chat_request_body_rejects_invalid_model():
    for bad_model in [None, "", "   ", 123]:
        with pytest.raises(HTTPException) as exc_info:
            rg.validate_chat_request_body({"model": bad_model, "messages": [{"role": "user", "content": "hi"}]})
        assert exc_info.value.status_code == 422
        assert "model must be a non-empty string" in exc_info.value.detail


def test_validate_chat_request_body_rejects_oversized_model_name():
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body({
            "model": "m" * 201,
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert exc_info.value.status_code == 422
    assert "model is too long" in exc_info.value.detail


def test_validate_chat_request_body_rejects_invalid_messages_list():
    for bad_msgs in [None, "", [], 123]:
        with pytest.raises(HTTPException) as exc_info:
            rg.validate_chat_request_body({"model": "gpt-5-mini", "messages": bad_msgs})
        assert exc_info.value.status_code == 422
        assert "messages must be a non-empty list" in exc_info.value.detail


def test_validate_chat_request_body_rejects_non_dict_message():
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body({
            "model": "gpt-5-mini",
            "messages": ["not a dict message"],
        })
    assert exc_info.value.status_code == 422
    assert "messages[0] must be an object" in exc_info.value.detail


def test_validate_chat_request_body_rejects_missing_or_empty_role():
    for bad_role in [None, "", "   ", 123]:
        with pytest.raises(HTTPException) as exc_info:
            rg.validate_chat_request_body({
                "model": "gpt-5-mini",
                "messages": [{"role": bad_role, "content": "hi"}],
            })
        assert exc_info.value.status_code == 422
        assert "messages[0].role must be a non-empty string" in exc_info.value.detail


def test_validate_chat_request_body_rejects_non_list_tools():
    for bad_tools in ["tool_string", 123, {"not": "a list"}]:
        with pytest.raises(HTTPException) as exc_info:
            rg.validate_chat_request_body({
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": bad_tools,
            })
        assert exc_info.value.status_code == 422
        assert "tools/functions must be a list" in exc_info.value.detail


def test_validate_chat_request_body_rejects_non_dict_tool():
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body({
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": ["not-a-dict-tool"],
        })
    assert exc_info.value.status_code == 422
    assert "tools[0] must be an object" in exc_info.value.detail


def test_validate_chat_request_body_rejects_non_json_serializable_tool():
    class _Unserializable:
        pass

    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body({
            "model": "gpt-5-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "fn": _Unserializable()}],
        })
    assert exc_info.value.status_code == 422
    assert "tools[0] is not JSON-serializable" in exc_info.value.detail


def test_validate_request_json_size_rejects_non_serializable():
    class _Unserializable:
        pass

    with pytest.raises(HTTPException) as exc_info:
        rg.validate_request_json_size({"unserializable": _Unserializable()})
    assert exc_info.value.status_code == 422
    assert "request body is not JSON-serializable" in exc_info.value.detail


# --- 4. 413 Error cases ---


def test_validate_chat_request_body_message_count_limit_413():
    body = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": f"msg {i}"} for i in range(5)],
    }
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body(body, max_messages=3)
    assert exc_info.value.status_code == 413
    assert "messages exceed limit (3)" in exc_info.value.detail


def test_validate_chat_request_body_message_content_length_limit_413():
    body = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "hello world"}],
    }
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body(body, max_message_text_chars=5)
    assert exc_info.value.status_code == 413
    assert "messages[0].content exceeds limit (5 chars)" in exc_info.value.detail


def test_validate_chat_request_body_tools_count_limit_413():
    body = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "function": {"name": f"tool_{i}"}}
            for i in range(3)
        ],
    }
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body(body, max_tools=2)
    assert exc_info.value.status_code == 413
    assert "tools exceed limit (2)" in exc_info.value.detail


def test_validate_chat_request_body_tool_schema_chars_limit_413():
    body = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"type": "function", "function": {"name": "oversized", "description": "x" * 100}}
        ],
    }
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_chat_request_body(body, max_tool_schema_chars=50)
    assert exc_info.value.status_code == 413
    assert "tools[0] exceeds limit (50 chars)" in exc_info.value.detail


def test_validate_request_json_size_limit_413():
    body = {"model": "gpt-5-mini", "payload": "x" * 100}
    with pytest.raises(HTTPException) as exc_info:
        rg.validate_request_json_size(body, max_chars=50)
    assert exc_info.value.status_code == 413
    assert "request body exceeds limit (50 chars)" in exc_info.value.detail


# --- 5. Rate limit and 429 tests ---


def test_rate_limit_bucket_formatting():
    req = _dummy_request("192.168.1.100")
    assert rg.rate_limit_bucket(req, "test-api-key-123456") == "192.168.1.100:test-api"

    req_no_client = SimpleNamespace(client=None)
    assert rg.rate_limit_bucket(req_no_client, "abcdefghij") == "unknown:abcdefgh"

    req_short_key = _dummy_request("10.0.0.1")
    assert rg.rate_limit_bucket(req_short_key, "abc") == "10.0.0.1:abc"


def test_rate_limit_disabled_when_zero_or_negative():
    req = _dummy_request("127.0.0.1")
    rg._rate_limit_buckets.clear()

    # limit <= 0 should never raise
    for _ in range(10):
        rg.enforce_chat_rate_limit(req, "test-key", rate_limit_per_minute=0)
        rg.enforce_chat_rate_limit(req, "test-key", rate_limit_per_minute=-1)


def test_rate_limit_exceeded_returns_429():
    req = _dummy_request("127.0.0.1")
    rg._rate_limit_buckets.clear()

    # Limit = 2
    rg.enforce_chat_rate_limit(req, "test-token", rate_limit_per_minute=2, rate_limit_window_seconds=60)
    rg.enforce_chat_rate_limit(req, "test-token", rate_limit_per_minute=2, rate_limit_window_seconds=60)

    with pytest.raises(HTTPException) as exc_info:
        rg.enforce_chat_rate_limit(req, "test-token", rate_limit_per_minute=2, rate_limit_window_seconds=60)
    assert exc_info.value.status_code == 429
    assert "rate limit exceeded" in exc_info.value.detail


# --- 6. Bucket sweep / TTL cleanup tests ---


def test_prune_rate_limit_buckets_purges_stale_and_empty():
    rg._rate_limit_buckets.clear()
    now = 1000.0
    window = 60.0  # Window is [940.0, 1000.0]

    # Bucket 1: all expired (< 940)
    rg._rate_limit_buckets["host1:token1"] = [850.0, 900.0, 930.0]
    # Bucket 2: completely empty
    rg._rate_limit_buckets["host2:token2"] = []
    # Bucket 3: mixed (one expired, one active)
    rg._rate_limit_buckets["host3:token3"] = [910.0, 950.0]
    # Bucket 4: all active
    rg._rate_limit_buckets["host4:token4"] = [970.0, 990.0]

    pruned = rg.prune_rate_limit_buckets(now=now, window_seconds=window)

    assert pruned == 2
    assert "host1:token1" not in rg._rate_limit_buckets
    assert "host2:token2" not in rg._rate_limit_buckets
    assert "host3:token3" in rg._rate_limit_buckets
    assert rg._rate_limit_buckets["host3:token3"] == [950.0]
    assert "host4:token4" in rg._rate_limit_buckets
    assert rg._rate_limit_buckets["host4:token4"] == [970.0, 990.0]


def test_enforce_chat_rate_limit_triggers_periodic_sweep(monkeypatch):
    rg._rate_limit_buckets.clear()
    rg._last_rate_limit_sweep = 100.0

    # Populate a stale bucket
    rg._rate_limit_buckets["stale-host:token"] = [50.0]

    # Make request at time 200.0 (100 seconds later, exceeding sweep_interval_seconds=60)
    monkeypatch.setattr(time, "time", lambda: 200.0)
    req = _dummy_request("active-host")
    rg.enforce_chat_rate_limit(req, "active-key", rate_limit_per_minute=10, rate_limit_window_seconds=60)

    # Stale bucket should have been pruned by the inline periodic sweep
    assert "stale-host:token" not in rg._rate_limit_buckets
    # Active bucket should exist
    assert "active-host:active-k" in rg._rate_limit_buckets


# --- 7. Server backward compatibility re-exports ---


def test_server_reexports_and_shared_dictionary():
    assert server._validate_chat_request_body is rg.validate_chat_request_body
    assert server._enforce_chat_rate_limit is rg.enforce_chat_rate_limit
    assert server._rate_limit_bucket is rg.rate_limit_bucket
    assert server._rate_limit_buckets is rg._rate_limit_buckets
    assert server._rate_limit_lock is rg._rate_limit_lock
