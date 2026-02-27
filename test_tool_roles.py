"""Test mit realistischer OpenClaw-History inkl. tool-Rollen."""
import json, urllib.request

PROXY = "http://127.0.0.1:11435/v1/chat/completions"
KEY   = "academicai-proxy-boku"

messages = [
    {"role": "system", "content": "Du bist Dagobert, ein hilfreicher Assistent."},
    {"role": "user", "content": "Schau mal was in der Datei steht."},
    {"role": "assistant", "content": "Ich lese die Datei.", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "t1", "content": "Dateiinhalt: Hallo Welt"},
    {"role": "assistant", "content": "Die Datei enthält: Hallo Welt."},
    {"role": "user", "content": "Danke. Was bedeutet das für das Projekt?"},
]

body = json.dumps({"model": "gpt-5", "messages": messages, "stream": True}).encode()
req = urllib.request.Request(PROXY, data=body,
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})

try:
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
        print(f"[OK]: {content[:120]}")
except urllib.error.HTTPError as e:
    print(f"[FAIL] HTTP {e.code}: {e.read().decode()[:150]}")
except Exception as e:
    print(f"[ERR]: {e}")
