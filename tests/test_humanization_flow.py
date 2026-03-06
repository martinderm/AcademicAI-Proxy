import asyncio

import server


def _run(coro):
    return asyncio.run(coro)


def test_humanization_pass_success():
    async def fake_completion(**kwargs):
        class Msg:
            content = "Natürliche Antwort ohne Meta."

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()

    # run_in_threadpool erwartet sync-callable; wir simulieren einfach direkt
    def fake_sync_completion(**kwargs):
        class Msg:
            content = "Natürliche Antwort ohne Meta."

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()

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


def test_humanization_prompt_contains_original_question_and_data():
    msgs = server._build_humanization_messages(
        original_user_query="Was weißt du über Zoom-Räume?",
        structured_content='{"rooms":[{"name":"Internal Room"}]}'
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "Return only the final answer text" in msgs[0]["content"]
    assert "Was weißt du über Zoom-Räume?" in msgs[1]["content"]
    assert '"rooms"' in msgs[1]["content"]


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
