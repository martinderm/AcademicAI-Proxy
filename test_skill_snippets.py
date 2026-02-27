import json
from pathlib import Path

import server


def test_skill_snippet_injection_for_mail_intent(tmp_path: Path):
    snippets = [
        {
            "id": "mailbox",
            "topics": ["mailbox", "email", "inbox"],
            "snippet": "Use Himalaya wrapper via exec.",
        }
    ]
    f = tmp_path / "snippets.json"
    f.write_text(json.dumps(snippets), encoding="utf-8")

    old_enabled = server.ENABLE_SKILL_SNIPPETS
    old_file = server.SKILL_SNIPPETS_FILE
    old_max = server.SKILL_SNIPPETS_MAX
    try:
        server.ENABLE_SKILL_SNIPPETS = True
        server.SKILL_SNIPPETS_FILE = str(f)
        server.SKILL_SNIPPETS_MAX = 1

        msgs = [{"role": "user", "content": "Was gibt's Neues in der Mailbox?"}]
        out = server._inject_skill_snippet_context(msgs, "Was gibt's Neues in der Mailbox?")

        assert len(out) == 2
        assert out[0]["role"] == "system"
        assert "SKILL CONTEXT" in out[0]["content"]
        assert "Himalaya" in out[0]["content"]
    finally:
        server.ENABLE_SKILL_SNIPPETS = old_enabled
        server.SKILL_SNIPPETS_FILE = old_file
        server.SKILL_SNIPPETS_MAX = old_max
