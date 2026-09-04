# System Map — Effects (Seiteneffekte, Grenzen & Umwelt)

> **Komponente**: Externe Einflüsse, Dateisystem-Mutationen, Ports und Sicherheitsgrenzen  
> **Kontext**: [`README.md`](README.md)

---

## 1. Externe Netzwerk-Aufrufe (Egress)

| Ziel | Protokoll | Zweck | Auth & Secrets |
| :--- | :--- | :--- | :--- |
| **BOKU AcademicAI Backend** | HTTPS (REST) | Ausführung der Chat-Completions | `X-Client-ID`, `X-Client-Secret` im Header |
| **Azure OpenAI Infrastructure** | HTTPS (via BOKU) | Prefix Caching & Inference | Indirekt über BOKU-Routing |

- **Timeout:** 120 Sekunden Standard-Timeout via `httpx.AsyncClient`.
- **Fehlerbehandlung:** HTTP 401/403/500 des BOKU-Backends werden in standardisierte OpenAI-Error-JSON-Payloads übersetzt ([`academicai/errors.py`](../../academicai/errors.py)).

---

## 2. Ports, Bindings & Prozess-Lifecycle

- **Standard Live-Port:** `127.0.0.1:11435`
  - Gesteuert über Umgebungsvariable `ACADEMICAI_PROXY_PORT`.
- **E2E-Test-Port:** `127.0.0.1:11436`
  - Strikt getrennt, damit Tests den produktiven Agentenbetrieb nicht unterbrechen.
- **Prozess-Steuerung (`server.pid`):**
  - Beim Start via [`start_server.ps1`](../../start_server.ps1) wird die Prozess-ID in `server.pid` hinterlegt.
  - [`stop_server.ps1`](../../stop_server.ps1) liest `server.pid` und beendet den Dienst sauber (`Stop-Process`).

---

## 3. Dateisystem-Mutationen (Schreiboperationen)

| Datei / Pfad | Auslöser | Verhalten |
| :--- | :--- | :--- |
| `server.log` | Jeder HTTP-Request | Tägliche Log-Rotation; automatische Bereinigung von Logs älter als 30 Tage. |
| `server.pid` | Start / Stop | Speichert die PID des laufenden Uvicorn-Prozesses. |
| `last_backend_request.json` | Bei `ACADEMICAI_DEBUG_DUMPS=true` | Überschreibt bei jedem Request den Rohpayload zum Debugging. |
| `skill_snippets.json` | Nach erfolgreichem Tool-Aufruf | Dynamisches Self-Learning: Ergänzt Heuristik-Tipps für erkannte Tools. |

---

## 4. Sicherheitsgrenzen & Hard Guardrails

1. **Klartext-Redaction ([`academicai/security.py`](../../academicai/security.py)):**  
   Alle Log-Ausgaben filtern Passwörter, API-Keys (`sk-...`), Client-Secrets und Authorization-Header heraus (`[REDACTED]`).
2. **Fast-Fail bei unsicheren Keys:**  
   Wenn `ACADEMICAI_PROXY_API_KEY` fehlt, kürzer als 16 Zeichen ist oder bekannte Standardwerte (`changeme`, `replace-with-strong-key`) enthält, verweigert der Server den Start mit Exit-Code 1.
3. **Mail-Destruction Guard (`_enforce_write_before_mail_delete`):**  
   Verhindert das versehentliche Löschen/Verschieben von E-Mails via Himalaya CLI (`exec`), falls die Batch-Operation nicht vorher explizit einen Schreibvorgang (`write`/`edit`) enthielt.
4. **Request- & Payload-Limits (413 / 422 Guardrails):**  
   Schützt Proxy und Backend vor Memory Exhaustion und unkontrollierten Payloads:
   - Tool-Obergrenze: `ACADEMICAI_MAX_TOOLS` (Default: 256)
   - Tool-Schema-Größe: `ACADEMICAI_MAX_TOOL_SCHEMA_CHARS` (Default: 100.000 Chars)
   - Message-Textlänge: `ACADEMICAI_MAX_MESSAGE_TEXT_CHARS` (Default: 200.000 Chars, in `.env` für 1M-Context-Modelle bis 1.000.000)
   - Gesamt-JSON-Payload: `ACADEMICAI_MAX_REQUEST_JSON_CHARS` (Default: 2.000.000 Chars, konfigurierbar bis 10.000.000)
   - Nachrichtenanzahl: `ACADEMICAI_MAX_MESSAGES` (Default: 300)
   - Bei Überschreitung wird der Request mit `413 Content Too Large` abgewiesen und mit exakten Zählwerten in `server.log` protokolliert.
