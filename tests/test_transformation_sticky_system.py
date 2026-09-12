from academicai.transformation import _normalize_messages


def test_system_context_survives_tail_trimming():
    # Build long history so trimming to last 20 applies.
    msgs = [{"role": "system", "content": "TOOL_SCHEMA_STICKY"}]
    for i in range(30):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"m{i}"})

    out = _normalize_messages(msgs)

    assert out
    assert out[0]["role"] == "user"
    assert "TOOL_SCHEMA_STICKY" in out[0]["content"]


def test_build_request_body_token_limit_auto_stripping():
    from academicai.transformation import build_request_body

    msgs = [{"role": "user", "content": "hello"}]
    params = {"max_tokens": 100, "max_completion_tokens": 100, "temperature": 0.7}

    # Gemini and o3 must have token limits stripped to avoid backend 500 error
    body_gemini = build_request_body("gemini-3.5-flash", msgs, params)
    assert "maxTokens" not in body_gemini
    assert "maxCompletionTokens" not in body_gemini
    assert body_gemini["temperature"] == 0.7

    body_o3 = build_request_body("o3", msgs, params)
    assert "maxTokens" not in body_o3
    assert "maxCompletionTokens" not in body_o3

    # gpt-4o and claude-opus-4-8 should keep maxTokens
    body_gpt4o = build_request_body("gpt-4o", msgs, params)
    assert body_gpt4o["maxTokens"] == 100

    body_claude = build_request_body("claude-opus-4-8", msgs, params)
    assert body_claude["maxTokens"] == 100


def test_build_request_body_reasoning_models_temperature():
    from academicai.transformation import build_request_body

    msgs = [{"role": "user", "content": "hello"}]
    params = {"temperature": 0.5}

    # Reasoning models require reasoningEffort="none" when temperature is set
    body_55 = build_request_body("gpt-5.5", msgs, params)
    assert body_55["reasoningEffort"] == "none"

    body_52 = build_request_body("gpt-5.2", msgs, params)
    assert body_52["reasoningEffort"] == "none"

    # Non-reasoning models or claude-opus-4-8 should not inject reasoningEffort
    body_claude = build_request_body("claude-opus-4-8", msgs, params)
    assert "reasoningEffort" not in body_claude

