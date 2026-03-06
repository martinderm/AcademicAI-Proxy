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
