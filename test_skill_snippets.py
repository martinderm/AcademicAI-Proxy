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


def test_auto_skill_learning_creates_snippet(tmp_path: Path):
    f = tmp_path / "snippets.json"
    f.write_text("[]\n", encoding="utf-8")

    old_file = server.SKILL_SNIPPETS_FILE
    old_enabled = server.ENABLE_AUTO_SKILL_LEARNING
    old_min_len = server.AUTO_SKILL_MIN_TOPIC_LEN
    old_topics_per_call = server.AUTO_SKILL_TOPICS_PER_CALL
    try:
        server.SKILL_SNIPPETS_FILE = str(f)
        server.ENABLE_AUTO_SKILL_LEARNING = True
        server.AUTO_SKILL_MIN_TOPIC_LEN = 4
        server.AUTO_SKILL_TOPICS_PER_CALL = 5

        server._learn_skill_snippets_from_tool_calls(
            "Bitte prüfe meine Mailbox auf neue WordPress Mails",
            [{"name": "exec", "arguments": {"command": "himalaya ..."}}],
        )

        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["id"] == "auto:exec"
        assert data[0]["source"] == "auto"
        assert "mailbox" in data[0]["topics"]
        assert data[0]["hits"] == 1
    finally:
        server.SKILL_SNIPPETS_FILE = old_file
        server.ENABLE_AUTO_SKILL_LEARNING = old_enabled
        server.AUTO_SKILL_MIN_TOPIC_LEN = old_min_len
        server.AUTO_SKILL_TOPICS_PER_CALL = old_topics_per_call


def test_auto_skill_learning_updates_existing_snippet(tmp_path: Path):
    snippets = [
        {
            "id": "auto:exec",
            "source": "auto",
            "hits": 2,
            "topics": ["mailbox"],
            "snippet": "Prefer exec first.",
        }
    ]
    f = tmp_path / "snippets.json"
    f.write_text(json.dumps(snippets), encoding="utf-8")

    old_file = server.SKILL_SNIPPETS_FILE
    old_enabled = server.ENABLE_AUTO_SKILL_LEARNING
    old_min_len = server.AUTO_SKILL_MIN_TOPIC_LEN
    try:
        server.SKILL_SNIPPETS_FILE = str(f)
        server.ENABLE_AUTO_SKILL_LEARNING = True
        server.AUTO_SKILL_MIN_TOPIC_LEN = 4

        server._learn_skill_snippets_from_tool_calls(
            "Bitte checke Newsletter und Gmail Inbox",
            [{"name": "exec", "arguments": {}}],
        )

        data = json.loads(f.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["id"] == "auto:exec"
        assert data[0]["hits"] == 3
        assert "newsletter" in data[0]["topics"]
        assert "gmail" in data[0]["topics"]
    finally:
        server.SKILL_SNIPPETS_FILE = old_file
        server.ENABLE_AUTO_SKILL_LEARNING = old_enabled
        server.AUTO_SKILL_MIN_TOPIC_LEN = old_min_len
