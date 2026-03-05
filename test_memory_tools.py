"""
Test: memory_search und memory_get via Tool-Emulation

Simuliert realistische OpenClaw-Requests mit echten Tool-Definitionen.
Testet:
  1. memory_search wird korrekt getriggert
  2. memory_get wird korrekt getriggert
  3. Kompletter Roundtrip: Tool-Call -> Tool-Result -> Text-Antwort
  4. Modell waehlt bei klarer Datei-Frage memory_get statt memory_search
"""

import json, time
import httpx

BASE = "http://127.0.0.1:11435"
HEADERS = {"Authorization": "Bearer test-proxy-key", "Content-Type": "application/json"}
MODEL = "gpt-5-mini"

# Realistische OpenClaw Tool-Definitionen (Subset, wie OpenClaw sie sendet)
OPENCLAW_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read the contents of a file. Supports text files and images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "offset": {"type": "number"},
                    "limit": {"type": "number"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Execute shell commands. Use for running scripts or system commands.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Brave Search API. Returns titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "count": {"type": "number"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Mandatory recall step: semantically search MEMORY.md and memory files before answering questions about prior work, decisions, dates, people, preferences, or todos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"},
                    "maxResults": {"type": "number"},
                    "minScore": {"type": "number"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_get",
            "description": "Safe snippet read from MEMORY.md or memory/*.md with optional from/lines; use after memory_search to pull only the needed lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path like MEMORY.md or memory/2026-02-25.md"},
                    "from": {"type": "number", "description": "Line number to start from"},
                    "lines": {"type": "number", "description": "Number of lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "session_status",
            "description": "Show current session status card (usage, time, cost, model). Use for model-use questions.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def post(messages, stream=False):
    return httpx.post(
        f"{BASE}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": OPENCLAW_TOOLS,
            "tool_choice": "auto",
            "stream": stream,
        },
        timeout=60,
    )


def ok(label, result, detail=""):
    status = "OK" if result else "FAIL"
    print(f"  [{status}] {label}" + (f"  ->  {detail}" if detail else ""))


# ── Test 1: memory_search getriggert ────────────────────────────────────────
print("\n" + "="*60)
print("Test 1: memory_search wird getriggert")
print("="*60)

r = post([{"role": "user", "content": "Was haben wir zuletzt ueber den AcademicAI Proxy entschieden?"}])
ok("HTTP 200", r.status_code == 200, f"got {r.status_code}")
if r.status_code == 200:
    d = r.json()
    choice = d["choices"][0]
    finish = choice["finish_reason"]
    tool_calls = choice["message"].get("tool_calls") or []
    ok("finish_reason = tool_calls", finish == "tool_calls", f"got {finish}")
    ok("tool_calls vorhanden", bool(tool_calls))
    if tool_calls:
        name = tool_calls[0]["function"]["name"]
        args = json.loads(tool_calls[0]["function"]["arguments"])
        ok("Tool = memory_search", name == "memory_search", f"got {name}")
        ok("query vorhanden", bool(args.get("query")), f"query='{args.get('query')}'")
        print(f"  Query: '{args.get('query')}'")
else:
    print(f"  Body: {r.text[:200]}")

time.sleep(2)

# ── Test 2: memory_get getriggert ────────────────────────────────────────────
print("\n" + "="*60)
print("Test 2: memory_get wird getriggert")
print("="*60)

r = post([{"role": "user", "content": "Lies die Datei MEMORY.md und zeig mir was drin steht."}])
ok("HTTP 200", r.status_code == 200, f"got {r.status_code}")
if r.status_code == 200:
    d = r.json()
    choice = d["choices"][0]
    finish = choice["finish_reason"]
    tool_calls = choice["message"].get("tool_calls") or []
    ok("finish_reason = tool_calls", finish == "tool_calls", f"got {finish}")
    if tool_calls:
        name = tool_calls[0]["function"]["name"]
        args = json.loads(tool_calls[0]["function"]["arguments"])
        ok("Tool = memory_get (oder read)", name in ("memory_get", "read"), f"got {name}")
        ok("path vorhanden", bool(args.get("path")), f"path='{args.get('path')}'")
        print(f"  Tool: {name}, path='{args.get('path')}'")

time.sleep(2)

# ── Test 3: Roundtrip memory_search -> Result -> Antwort ─────────────────────
print("\n" + "="*60)
print("Test 3: Roundtrip memory_search -> Tool-Result -> Text")
print("="*60)

messages = [
    {"role": "user", "content": "Was weisst du ueber den AcademicAI Proxy aus dem Memory?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_mem001",
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "arguments": '{"query": "AcademicAI Proxy"}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_mem001",
        "content": json.dumps([
            {"score": 0.95, "path": "MEMORY.md#12", "text": "AcademicAI Proxy: FastAPI auf Port 11435, OpenAI-kompatibel, API-Key test-proxy-key. Auto-Start via Scheduled Task."},
            {"score": 0.88, "path": "memory/2026-02-24.md#45", "text": "Tool-Call-Emulation implementiert via JSON-Mode (Strategie A). Alle 4 Tests gruen."},
        ], ensure_ascii=False),
    },
]

r = post(messages)
ok("HTTP 200", r.status_code == 200, f"got {r.status_code}")
if r.status_code == 200:
    d = r.json()
    choice = d["choices"][0]
    finish = choice["finish_reason"]
    content = choice["message"].get("content") or ""
    tool_calls = choice["message"].get("tool_calls") or []
    ok("finish_reason = stop", finish == "stop", f"got {finish}")
    ok("Text-Antwort vorhanden", bool(content), f"'{content[:100]}'")
    ok("Kein weiterer tool_call", not tool_calls)
    if content:
        print(f"  Antwort: '{content[:200]}'")
else:
    print(f"  Body: {r.text[:300]}")

time.sleep(2)

# ── Test 4: memory_get mit from/lines ────────────────────────────────────────
print("\n" + "="*60)
print("Test 4: memory_get mit from/lines Parametern")
print("="*60)

r = post([{"role": "user", "content": "Lies die ersten 10 Zeilen aus memory/2026-02-24.md"}])
ok("HTTP 200", r.status_code == 200, f"got {r.status_code}")
if r.status_code == 200:
    d = r.json()
    choice = d["choices"][0]
    finish = choice["finish_reason"]
    tool_calls = choice["message"].get("tool_calls") or []
    ok("finish_reason = tool_calls", finish == "tool_calls", f"got {finish}")
    if tool_calls:
        name = tool_calls[0]["function"]["name"]
        args = json.loads(tool_calls[0]["function"]["arguments"])
        ok("Tool = memory_get (oder read)", name in ("memory_get", "read"), f"got {name}")
        ok("path enthaelt 2026-02-24", "2026-02-24" in str(args.get("path", "")), f"path='{args.get('path')}'")
        lines_arg = args.get("lines") or args.get("limit")
        ok("lines/limit ~10", lines_arg is not None, f"lines={lines_arg}")
        print(f"  Tool: {name}, args: {args}")

print("\n" + "="*60)
print("Tests abgeschlossen.")
print("="*60)
