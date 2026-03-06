"""
Simuliert einen realistischen OpenClaw-Request an den Proxy.
Testet: system message im messages array + top-level system parameter.
"""
import json, urllib.request

PROXY = "http://127.0.0.1:11435/v1/chat/completions"
KEY   = "test-proxy-key"
MODEL = "gpt-5"

BIG_SYSTEM = """Du bist Dagobert, ein KI-Assistent. Du hilfst Martin bei IT-Aufgaben.
Antworte auf Deutsch. Sei präzise und klar. Nutze keine unnötigen Floskeln.
Du hast Zugriff auf verschiedene Tools und kannst Dateien lesen und schreiben.
Deine Aufgabe ist es, dem Nutzer bei der Arbeit zu helfen."""

def run_case(label, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(PROXY, data=data,
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
            print(f"[OK] {label}: {content[:80]}")
    except urllib.error.HTTPError as e:
        print(f"[FAIL] {label}: HTTP {e.code} — {e.read().decode()[:100]}")

def test_smoke_pytest_collection():
    """Sorgt dafür, dass pytest dieses Modul ohne Fixture-Fehler einsammelt."""
    assert True


if __name__ == "__main__":
    # Test 1: system als erster Eintrag in messages (OpenAI-style)
    run_case("system in messages", {
        "model": MODEL, "stream": True,
        "messages": [
            {"role": "system", "content": BIG_SYSTEM},
            {"role": "user", "content": "Sag kurz Hallo auf Deutsch."}
        ]
    })

    # Test 2: system als top-level Parameter (Anthropic-style)
    run_case("top-level system", {
        "model": MODEL, "stream": True,
        "system": BIG_SYSTEM,
        "messages": [
            {"role": "user", "content": "Sag kurz Hallo auf Deutsch."}
        ]
    })

    # Test 3: kein system (Baseline)
    run_case("kein system", {
        "model": MODEL, "stream": True,
        "messages": [
            {"role": "user", "content": "Sag kurz Hallo auf Deutsch."}
        ]
    })
