"""
Test aller AcademicAI-Modelle via Proxy — exaktes OpenClaw-Request-Format.

Basiert auf last_request.json (echter OpenClaw-Request von sample-agent).
Keys die OpenClaw schickt: model, messages, stream, stream_options, store, max_completion_tokens, tools
- tools, stream_options, store werden vom Proxy gefiltert
- content als list wird in text extrahiert
- system/tool/consecutive-roles werden normalisiert
"""
import json, urllib.request

PROXY = "http://127.0.0.1:11435/v1/chat/completions"
KEY   = "test-proxy-key"

SYSTEM = "Du bist ein hilfreicher KI-Assistent fuer allgemeine Aufgaben."
MODELS = ["gpt-4o", "o3", "gpt-5-nano", "gpt-5-mini", "gpt-5", "Mistral-Large-3"]

# Realistisches OpenClaw-Format inkl. aller problematischen Felder
def make_openclaw_request(model: str) -> dict:
    return {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},  # wird vom Proxy gefiltert
        "store": False,                               # wird vom Proxy gefiltert
        "max_completion_tokens": 32000,
        "tools": [                                    # wird vom Proxy gefiltert
            {"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}
        ],
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Sag Hallo auf Deutsch in einem Satz."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "read", "arguments": '{"path": "test.md"}'}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "Dateiinhalt: Willkommen"}
            ]},
            {"role": "assistant", "content": "Ich habe die Datei gelesen."},
            {"role": "user", "content": "Danke."},
        ]
    }

def call_proxy(body: dict) -> str:
    data = json.dumps(body).encode()
    req = urllib.request.Request(PROXY, data=data,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        content = ""
        for line in raw.splitlines():
            if line.startswith("data:") and "[DONE]" not in line:
                try:
                    chunk = json.loads(line[5:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta: content += delta
                except: pass
        return content

import time

print("=== OpenClaw-Format Test — alle Modelle ===\n")
ok = fail = 0
for i, m in enumerate(MODELS):
    if i > 0:
        time.sleep(5)  # Rate-Limit schonen
    body = make_openclaw_request(m)
    body["model"] = m
    try:
        content = call_proxy(body)
        print(f"  [OK] {m}: {content[:80]}")
        ok += 1
    except urllib.error.HTTPError as e:
        print(f"  [FAIL] {m}: HTTP {e.code} — {e.read().decode()[:100]}")
        fail += 1
    except Exception as e:
        print(f"  [ERR] {m}: {type(e).__name__}: {str(e)[:80]}")
        fail += 1

print(f"\nErgebnis: {ok}/{ok+fail} OK")
