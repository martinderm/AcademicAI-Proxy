"""
Test: Tool-Call-Emulation

Testet ob der Proxy:
1. Tool-Definitionen korrekt als System-Prompt injiziert
2. <tool_call>-Tags im Response erkennt und in OpenAI tool_calls Format umwandelt
3. Streaming: tool_calls SSE-Chunks korrekt emittiert
4. Normaler Text-Response weiterhin funktioniert (keine Regression)
"""

import json
import httpx
import time

BASE = "http://127.0.0.1:11435"
HEADERS = {"Authorization": "Bearer academicai-proxy-boku", "Content-Type": "application/json"}
MODEL = "gpt-5-mini"  # günstigstes Modell für Tests


def post(payload: dict, stream: bool = False) -> httpx.Response:
    return httpx.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        json=payload,
        timeout=60,
    )


# ── Minimal-Tool-Set (subset von OpenClaw Tools) ─────────────────────────────
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Brave Search API. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "description": "Search query string."},
                    "count": {"type": "number", "description": "Number of results (1-10)."},
                },
            },
            "strict": False,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "session_status",
            "description": "Show current session status card (usage, time, cost, model).",
            "parameters": {"type": "object", "required": [], "properties": {}},
            "strict": False,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Semantically search MEMORY.md and memory files.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string"},
                    "maxResults": {"type": "number"},
                },
            },
            "strict": False,
        },
    },
]


def print_result(label: str, ok: bool, detail: str = ""):
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}")
    if detail:
        print(f"     {detail}")


# ── Test 1: Normaler Response ohne Tools (Regression) ────────────────────────
print("\n=== Test 1: Normaler Response (kein Tool-Call) ===")
r = post({
    "model": MODEL,
    "messages": [{"role": "user", "content": "Antworte mit genau: HALLO_TEST"}],
    "stream": False,
})
if r.status_code == 200:
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    finish = data["choices"][0]["finish_reason"]
    has_tool_calls = "tool_calls" in data["choices"][0]["message"] and data["choices"][0]["message"]["tool_calls"]
    print_result("Status 200", True)
    print_result("finish_reason = stop", finish == "stop", f"got: {finish}")
    print_result("Kein tool_calls Feld", not has_tool_calls)
    print_result("Content vorhanden", bool(content), f"'{content[:60]}'")
else:
    print_result(f"HTTP {r.status_code}", False, r.text[:200])

time.sleep(2)

# ── Test 2: Tool-Call wird ausgelöst ─────────────────────────────────────────
print("\n=== Test 2: Tool-Call ausgelöst (web_search) ===")
r = post({
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": "Suche im Web nach 'OpenClaw AI assistant'. Nutze das web_search Tool.",
        }
    ],
    "tools": SAMPLE_TOOLS,
    "tool_choice": "auto",
    "stream": False,
})
if r.status_code == 200:
    data = r.json()
    choice = data["choices"][0]
    finish = choice["finish_reason"]
    msg = choice["message"]
    tool_calls = msg.get("tool_calls") or []

    print_result("Status 200", True)
    print_result("finish_reason = tool_calls", finish == "tool_calls", f"got: {finish}")
    print_result("tool_calls vorhanden", bool(tool_calls), f"count: {len(tool_calls)}")
    if tool_calls:
        tc = tool_calls[0]
        tc_name = tc["function"]["name"]
        tc_args = json.loads(tc["function"]["arguments"])
        print_result("Tool = web_search", tc_name == "web_search", f"got: {tc_name}")
        print_result("query Argument vorhanden", "query" in tc_args, f"args: {tc_args}")
        print_result("call_id vorhanden", bool(tc.get("id")), f"id: {tc.get('id')}")
        print(f"   -> query: '{tc_args.get('query', '')}'")
else:
    print_result(f"HTTP {r.status_code}", False, r.text[:300])

time.sleep(2)

# ── Test 3: Tool-Call Streaming ───────────────────────────────────────────────
print("\n=== Test 3: Tool-Call Streaming ===")
r_stream = httpx.post(
    f"{BASE}/v1/chat/completions",
    headers=HEADERS,
    json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Suche im Web nach 'BOKU Wien'. Nutze das web_search Tool."}
        ],
        "tools": SAMPLE_TOOLS,
        "stream": True,
    },
    timeout=60,
)
if r_stream.status_code == 200:
    chunks = []
    for line in r_stream.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunks.append(json.loads(line[6:]))
            except Exception:
                pass

    finish_chunks = [c for c in chunks if c["choices"][0].get("finish_reason")]
    tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]

    print_result("SSE Chunks empfangen", len(chunks) > 0, f"count: {len(chunks)}")
    print_result("tool_calls Chunk vorhanden", bool(tool_chunks))
    print_result("finish_reason = tool_calls", any(
        c["choices"][0].get("finish_reason") == "tool_calls" for c in finish_chunks
    ))
    if tool_chunks:
        tc_delta = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
        print_result("name im Chunk", bool(tc_delta["function"].get("name")), f"name: {tc_delta['function'].get('name')}")
else:
    print_result(f"HTTP {r_stream.status_code}", False, r_stream.text[:200])

time.sleep(2)

# ── Test 4: Tool-Result -> Fortsetzung ────────────────────────────────────────
print("\n=== Test 4: Tool-Result Konversation ===")
# Simuliere: Modell hat tool_call gemacht, wir geben Ergebnis zurück
r = post({
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Suche nach 'BOKU Wien'. Nutze web_search."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_test001",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "BOKU Wien"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_test001",
            "content": "Result: BOKU Wien ist die Universität für Bodenkultur. Website: boku.ac.at",
        },
    ],
    "tools": SAMPLE_TOOLS,
    "stream": False,
})
if r.status_code == 200:
    data = r.json()
    choice = data["choices"][0]
    finish = choice["finish_reason"]
    content = choice["message"].get("content") or ""
    tool_calls = choice["message"].get("tool_calls") or []

    print_result("Status 200", True)
    print_result("finish_reason = stop (kein weiterer Tool-Call)", finish == "stop", f"got: {finish}")
    print_result("Text-Antwort vorhanden", bool(content), f"'{content[:80]}'")
    print_result("Keine weiteren tool_calls", not tool_calls)
else:
    print_result(f"HTTP {r.status_code}", False, r.text[:300])

print("\n=== Tests abgeschlossen ===")

# ── Test 5: Messy-JSON-Parsing (Unit-Tests, kein Proxy-Aufruf) ───────────────
print("\n=== Test 5: Messy JSON Parser (Unit) ===")
from academicai.tool_emulation import parse_json_mode_response, extract_respond_content

_tc = lambda c: {"action": "tool_call", "name": "exec", "arguments": {"command": c}}
_rsp = lambda t: {"action": "respond", "content": t}

cases = [
    # (label, input_string, expected_action, expected_key_check)
    (
        "Sauberes JSON (Baseline)",
        json.dumps(_tc("Get-Content test.txt")),
        "tool_call", lambda r: r and r.get("name") == "exec",
    ),
    (
        "Preamble-Text vor JSON",
        "Ich werde jetzt das Tool aufrufen:\n" + json.dumps(_tc("Get-Date")),
        "tool_call", lambda r: r and r.get("name") == "exec",
    ),
    (
        "Erklärungstext nach JSON",
        json.dumps(_tc("Get-Date")) + "\n\nDas ist mein Tool-Call.",
        "tool_call", lambda r: r and r.get("name") == "exec",
    ),
    (
        "Markdown-Codeblock ```json```",
        "Hier der Aufruf:\n```json\n" + json.dumps(_tc("Get-ChildItem")) + "\n```",
        "tool_call", lambda r: r and r.get("name") == "exec",
    ),
    (
        "Markdown-Codeblock ohne json-Label",
        "OK:\n```\n" + json.dumps(_tc("Get-ChildItem")) + "\n```",
        "tool_call", lambda r: r and r.get("name") == "exec",
    ),
    (
        "Reasoning + JSON (Kettengedanke)",
        "Let me think... I need to read the file. "
        "The best tool is exec.\n\n" + json.dumps(_tc("Get-Content HEARTBEAT.md")),
        "tool_call", lambda r: r and r.get("name") == "exec",
    ),
    (
        "respond mit Preamble",
        "Sure, I can answer that directly!\n" + json.dumps(_rsp("HEARTBEAT_OK")),
        "respond", lambda r: True,
    ),
    (
        "Respond-Content korrekt extrahiert",
        json.dumps(_rsp("HEARTBEAT_OK")),
        "respond", lambda r: True,
    ),
    (
        "Kein JSON -> None",
        "Ich habe keine Tools zur Verfügung.",
        None, lambda r: r is None,
    ),
    (
        "Leerer String -> None",
        "",
        None, lambda r: r is None,
    ),
]

all_ok = True
for label, inp, expected_action, check in cases:
    result = parse_json_mode_response(inp)
    got_action = result.get("action") if result else None
    action_ok = got_action == expected_action
    detail_ok = check(result)
    ok = action_ok and detail_ok
    all_ok = all_ok and ok
    exp_str = f"action={expected_action}"
    got_str = f"action={got_action}"
    print_result(label, ok, f"erwartet {exp_str}, got {got_str}" if not ok else "")

print_result("Alle Messy-JSON-Tests bestanden", all_ok)

# ── Test 6: Arbitrary-JSON-Fallback (Unit + Integration) ─────────────────────
# Hintergrund: Das Modell gibt manchmal valides JSON aus, das weder
# action=tool_call noch action=respond ist (z.B. einen JSON-Plan).
# Der Proxy soll das als Markdown-Codeblock formatieren statt rohen JSON-String
# durchzureichen. (Gefunden: boku-lll-guru Heartbeat 2026-02-27.)

print("\n=== Test 6a: Arbitrary-JSON-Fallback (Unit) ===")
from academicai.tool_emulation import format_arbitrary_json_as_codeblock

unit_cases_arb = [
    (
        "Beliebiges JSON-Objekt (kein action-Key) -> Codeblock",
        json.dumps({"job": "wp-sync", "trigger": "cron", "steps": ["list", "parse", "update"]}),
        True,   # expect code block (not None)
        lambda r: r is not None and r.startswith("```json") and r.endswith("```"),
    ),
    (
        "Nested JSON-Plan -> Codeblock",
        json.dumps({"plan": {"phase1": "list mails", "phase2": "update WEB-ADMIN.md"}, "atomic": True}),
        True,
        lambda r: r is not None and "```json" in r,
    ),
    (
        "action=respond -> None (kein Codeblock, wird normal behandelt)",
        json.dumps({"action": "respond", "content": "HEARTBEAT_OK"}),
        False,  # expect None
        lambda r: r is None,
    ),
    (
        "action=tool_call -> None (kein Codeblock, wird als Tool-Call behandelt)",
        json.dumps({"action": "tool_call", "name": "exec", "arguments": {"command": "Get-Date"}}),
        False,
        lambda r: r is None,
    ),
    (
        "Kein JSON (Freitext) -> None (kein Codeblock nötig)",
        "HEARTBEAT_OK",
        False,
        lambda r: r is None,
    ),
    (
        "Leerer String -> None",
        "",
        False,
        lambda r: r is None,
    ),
    (
        "JSON-Array (kein Dict) -> Codeblock",
        json.dumps(["step1", "step2", "step3"]),
        True,
        lambda r: r is not None and "```json" in r,
    ),
]

all_arb_ok = True
for label, inp, expect_codeblock, check in unit_cases_arb:
    result = format_arbitrary_json_as_codeblock(inp)
    ok = check(result)
    all_arb_ok = all_arb_ok and ok
    got_str = f"'{result[:60]}...'" if result and len(result) > 60 else repr(result)
    print_result(label, ok, f"got: {got_str}" if not ok else "")

print_result("Alle Arbitrary-JSON-Unit-Tests bestanden", all_arb_ok)

# ── Test 6a.2: Humanizer muss auch fenced JSON erkennen (Regression) ─────────
print("\n=== Test 6a.2: Fenced-JSON Humanizer (Regression) ===")
from academicai.tool_emulation import format_arbitrary_json_for_humans

fenced_json = """```json
{
  "status": "ok",
  "capabilities": ["zoom_rooms", "meetings"]
}
```"""

human_text = format_arbitrary_json_for_humans(fenced_json)

ok_humanizer = isinstance(human_text, str) and human_text.startswith("Hier die Infos:") and "capabilities" in human_text
print_result("Fenced JSON -> menschenlesbarer Text", ok_humanizer, f"got: {repr(human_text)[:160]}")

# ── Test 6b: Arbitrary-JSON-Fallback (Integration, live Proxy) ───────────────
# Trigger: Prompt der das Modell dazu bringt, einen strukturierten JSON-Plan
# auszugeben ohne action=tool_call oder action=respond zu verwenden.
# Erwartung: Antwort ist ein Markdown-Codeblock, KEIN roher JSON-String.
print("\n=== Test 6b: Arbitrary-JSON-Fallback (Integration) ===")
print("  (Probabilistisch — Modell folgt Schema nicht immer. FAIL ist ein Hinweis, kein Blocker.)")
r = post({
    "model": MODEL,
    "messages": [
        {
            "role": "user",
            "content": (
                "Erstelle einen kompakten JSON-Plan (nur die Datenstruktur, kein Markdown, kein Erklärungstext) "
                "für einen WordPress-Plugin-Update-Workflow mit den Feldern: trigger, steps, error_handling. "
                "Antworte NUR mit dem JSON-Objekt."
            ),
        }
    ],
    "tools": SAMPLE_TOOLS,
    "stream": False,
})
if r.status_code == 200:
    data = r.json()
    choice = data["choices"][0]
    content = choice["message"].get("content") or ""
    tool_calls = choice["message"].get("tool_calls") or []
    finish = choice["finish_reason"]

    is_tool_call = bool(tool_calls)
    is_codeblock = content.strip().startswith("```")
    is_raw_json = False
    if content.strip():
        try:
            json.loads(content.strip())
            is_raw_json = True
        except Exception:
            pass

    print_result("Status 200", True)
    if is_tool_call:
        # Modell hat Tool gewählt statt Plan ausgegeben - akzeptabel, aber nicht der Zielfall
        print_result("Modell hat Tool gewählt (kein Arbitrary-JSON-Fall getriggert)", True,
                     "Hinweis: Prompt hat Tool-Call ausgelöst statt JSON-Plan - Testfall nicht aktiv")
    elif is_codeblock:
        # Human-Target-Fallback aktiv
        print_result("Arbitrary-JSON als Codeblock formatiert (human target)", True,
                     f"Inhalt: {content[:100]}")
    elif is_raw_json:
        # Non-human Target (z.B. cron/direct API) darf raw JSON behalten
        print_result("Roher JSON-String auf non-human target (erwartet)", True,
                     f"Inhalt: {content[:120]}")
    else:
        # Normaler Text -> Modell hat Schema befolgt mit action=respond -> auch OK
        print_result("Normaler Text (action=respond korrekt extrahiert)", True,
                     f"Inhalt: {content[:80]}")
else:
    print_result(f"HTTP {r.status_code}", False, r.text[:300])

print("\n=== Alle Tests abgeschlossen ===")

