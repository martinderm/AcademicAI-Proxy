"""
Feasibility-Test: Strategie A — JSON-Mode Tool-Emulation

Phase 1:
  1. Prüfen ob response_format: {type: "json_object"} vom Backend akzeptiert wird
  2. Prüfen ob Modell zuverlässig {"action": "tool_call", ...} produziert
  3. Edge Cases: kein Tool nötig, Grenzfall, mehrere mögliche Tools

Direkt gegen den Proxy (der response_format bereits durchreicht).
"""

import json
import httpx
import time

BASE = "http://127.0.0.1:11435"
HEADERS = {"Authorization": "Bearer academicai-proxy-boku", "Content-Type": "application/json"}
MODEL = "gpt-5-mini"

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web. Use when current/external information is needed.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search the user's personal memory files.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Run a shell command on the local machine.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string"},
                },
            },
        },
    },
]


SYSTEM_PROMPT_TEMPLATE = """\
You have access to the following tools. When the user's request requires a tool, you MUST use it.

Tools:
{tool_list}

You MUST respond with a valid JSON object. Choose exactly one:
- If a tool is needed: {{"action": "tool_call", "name": "<tool_name>", "arguments": {{...}}}}
- If you can answer directly: {{"action": "respond", "content": "<your answer>"}}

No explanation, no markdown, no extra text. Only the JSON object.
"""

def build_tool_list(tools):
    lines = []
    for t in tools:
        fn = t["function"]
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        param_str = ", ".join(
            f"{p}{'?' if p not in required else ''}: {v.get('type', 'any')}"
            for p, v in params.items()
        )
        lines.append(f"- {fn['name']}({param_str}) — {fn['description']}")
    return "\n".join(lines)


def post_json_mode(user_msg: str, tools=None) -> dict:
    tool_list = build_tool_list(tools or SAMPLE_TOOLS)
    system = SYSTEM_PROMPT_TEMPLATE.format(tool_list=tool_list)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }

    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )
    return r


def ok(label, result, detail=""):
    status = "OK" if result else "FAIL"
    arrow = " -> " if detail else ""
    print(f"  [{status}] {label}{arrow}{detail}")


def run_test(label, user_msg, expect_action, expect_tool=None, tools=None):
    print(f"\n{'-'*60}")
    print(f"Test: {label}")
    print(f"User: \"{user_msg}\"")

    r = post_json_mode(user_msg, tools)

    ok("HTTP 200", r.status_code == 200, f"got {r.status_code}")
    if r.status_code != 200:
        print(f"  Body: {r.text[:300]}")
        return

    data = r.json()
    raw_content = data["choices"][0]["message"]["content"] or ""
    print(f"  Raw: {raw_content[:200]}")

    # JSON parsen
    try:
        parsed = json.loads(raw_content)
        ok("Valid JSON", True)
    except json.JSONDecodeError as e:
        ok("Valid JSON", False, str(e))
        return

    action = parsed.get("action")
    ok(f"action = '{expect_action}'", action == expect_action, f"got '{action}'")

    if expect_action == "tool_call":
        name = parsed.get("name")
        args = parsed.get("arguments", {})
        ok(f"name = '{expect_tool}'", name == expect_tool, f"got '{name}'")
        ok("arguments ist dict", isinstance(args, dict), f"got {type(args).__name__}")
        if args:
            print(f"  Arguments: {json.dumps(args, ensure_ascii=False)}")
    elif expect_action == "respond":
        content = parsed.get("content", "")
        ok("content vorhanden", bool(content), f"'{content[:80]}'")


# ── Test 1: response_format wird akzeptiert (einfachster Fall) ───────────────
print("=" * 60)
print("STRATEGIE A — JSON-Mode Feasibility Tests")
print(f"Modell: {MODEL}")
print("=" * 60)

run_test(
    label="1. Klarer Tool-Call (web_search)",
    user_msg="Was ist das Wetter heute in Wien?",
    expect_action="tool_call",
    expect_tool="web_search",
)
time.sleep(2)

run_test(
    label="2. Klarer Tool-Call (exec)",
    user_msg="Zeig mir den Inhalt des Ordners C:\\Users",
    expect_action="tool_call",
    expect_tool="exec",
)
time.sleep(2)

run_test(
    label="3. Kein Tool nötig — direkte Antwort",
    user_msg="Was ist 7 mal 8?",
    expect_action="respond",
)
time.sleep(2)

run_test(
    label="4. Kein Tool nötig — Wissens-Frage",
    user_msg="Was ist die Hauptstadt von Österreich?",
    expect_action="respond",
)
time.sleep(2)

run_test(
    label="5. Grenzfall — könnte Tool sein oder nicht",
    user_msg="Was weißt du über die BOKU Wien?",
    expect_action=None,  # beides akzeptabel
)
time.sleep(2)

run_test(
    label="6. memory_search",
    user_msg="Was haben wir zuletzt über den AcademicAI Proxy besprochen?",
    expect_action="tool_call",
    expect_tool="memory_search",
)
time.sleep(2)

run_test(
    label="7. Nur ein Tool verfügbar — muss es nutzen",
    user_msg="Suche im Web nach OpenClaw.",
    expect_action="tool_call",
    expect_tool="web_search",
    tools=[SAMPLE_TOOLS[0]],  # nur web_search
)

print(f"\n{'='*60}")
print("Tests abgeschlossen.")
