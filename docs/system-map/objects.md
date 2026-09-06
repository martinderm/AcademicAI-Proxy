# System Map — Objects (Nomen, Schemas & State)

> **Komponente**: Datenstrukturen, Schnittstellenverträge und Zustandsobjekte  
> **Kontext**: [`README.md`](README.md)

---

## 1. Inbound Request Schemas (Client $\to$ Proxy)

### `POST /v1/chat/completions`
Standard-OpenAI-Payload:
- **`model`** `(str, required)`: Zielmodell (z. B. `gpt-4o`, `gpt-4o-mini`, `academicai-default`).
- **`messages`** `(list[dict], required)`: Liste von Rollen-Objekten:
  - `role`: `system` | `user` | `assistant` | `tool`
  - `content`: `str` oder `list[dict]` (z. B. multimodaler Content / Text-Parts).
  - `tool_calls` `(optional, list)`: Vorherige Tool-Aufrufe bei Assistant-Nachrichten.
  - `tool_call_id` `(optional, str)`: Referenz-ID bei `role: tool`.
- **`tools`** `(list[dict], optional)`: OpenAI-Funktionsdefinitionen:
  - `type`: `"function"`
  - `function`: `{ "name": str, "description": str, "parameters": dict }`
- **`stream`** `(bool, default: False)`: SSE-Streaming aktivieren (`text/event-stream`).
- **`temperature`**, **`max_tokens`**, **`top_p`** `(optional)`: Generierungsparameter.

### `GET /v1/models`
Liefert verfügbare Modelle als OpenAI ModelList:
```json
{
  "object": "list",
  "data": [
    { "id": "gpt-4o", "object": "model", "owned_by": "academicai" },
    { "id": "gpt-4o-mini", "object": "model", "owned_by": "academicai" }
  ]
}
```

---

## 2. Outbound Backend Schemas (Proxy $\to$ BOKU AcademicAI)

Das BOKU-Backend verlangt spezifische Auth-Header und REST-Strukturen ([`academicai/auth.py`](../../academicai/auth.py)):
- **Headers**:
  - `X-Client-ID`: Client-ID aus Umgebungsvariable `ACADEMICAI_CLIENT_ID`.
  - `X-Client-Secret`: Secret aus `ACADEMICAI_CLIENT_SECRET`.
  - `Content-Type`: `application/json`.
- **Payload**:
  - Zusammengeführte Prompt-Struktur, optimiert für das Azure-Prefix-Caching (System-Instruktionen am Kopf der ersten User-Message).

---

## 3. Emulations- & Transformationsobjekte

### Message-Rollen-Normalisierung ([`academicai/transformation.py`](../../academicai/transformation.py))
Das BOKU-Backend unterstützt ausschließlich die Rollen **`user`** und **`assistant`**. Eingehende OpenAI-Rollen werden daher deterministisch transformiert:

| Eingehende OpenAI-Rolle | Zielrolle am BOKU-Backend | Transformationsregel |
| :--- | :--- | :--- |
| `system` | `user` | Inhalt wird an den Beginn der allerersten `user`-Message eingefügt (Sticky System Message / Azure Prefix Caching). |
| `tool` | `user` | Formatiert als `[Tool result (id: <tool_call_id>)]\n<content>`. |
| `assistant` (mit `tool_calls`) | `assistant` | Formatiert als `[Tool call: <name>(<arguments>)]`. |
| Aufeinanderfolgende gleiche Rollen | `user`/`assistant` | Werden zu einer einzigen Nachricht mit `\n\n` zusammengeführt. |

### `ToolCall` (OpenAI-konform synthetisiert)
Wird von [`academicai/tool_emulation.py`](../../academicai/tool_emulation.py) aus dem Modell-Freitext erzeugt:
```json
{
  "id": "call_abc123456",
  "type": "function",
  "function": {
    "name": "exec",
    "arguments": "{\"command\": \"dir\"}"
  }
}
```

### Interne Provider-Dataclasses ([`academicai/provider.py`](../../academicai/provider.py))
- `Message`: `role: str`, `content: str`
- `Choice`: `index: int`, `message: Message`, `finish_reason: str`
- `Usage`: `prompt_tokens: int`, `completion_tokens: int`, `total_tokens: int`
- `CompletionResponse`: `id: str`, `object: str`, `created: int`, `model: str`, `choices: list[Choice]`, `usage: Usage`

### `SkillSnippet` (Zustandsobjekt für Self-Learning)
In `skill_snippets.json` persistierte Heuristik-Tipps:
```json
{
  "id": "himalaya-envelope-list",
  "topics": ["mail", "envelope", "himalaya"],
  "snippet": "Use envelope list -s 50 to search mailboxes efficiently."
}
```

---

## 4. Konfiguration & Runtime State ([`academicai/config.py`](../../academicai/config.py))

Das Modul [`academicai/config.py`](../../academicai/config.py) ist die zentrale **Single Source of Truth (SSOT)** für alle Laufzeitkonfigurationen, Umgebungsvariablen, Standardwerte, Schutzgrenzen und Validierungen des Proxies.

### Entkoppelte Initialisierung & Startup-Validierung
- **Kein Crash beim Import:** Das reine Importieren von `academicai.config` oder `server` löst keine fatalen Validierungsfehler (`RuntimeError`) aus, selbst wenn Umgebungsvariablen nicht gesetzt sind oder Standard-Testschlüssel verwendet werden.
- **Explizite Validierung (`validate_config()`):** Die Integritätsprüfung des Proxy-API-Keys (`_validate_proxy_api_key`) wird gezielt beim Anwendungsstart (`@app.on_event("startup")` bzw. `if __name__ == "__main__":`) oder explizit über `validate_config()` aufgerufen.
- **Rückwärtskompatibilität:** `server.py` re-exportiert alle Konfigurationskonstanten sowie `_validate_proxy_api_key`, sodass bestehende Tests und externe Aufrufer ohne Änderungen weiterfunktionieren.

### Konfigurations-Schema & Defaults

| Variable / Konstante | Env-Variable | Typ | Default | Zweck |
| :--- | :--- | :--- | :--- | :--- |
| `PORT` | `ACADEMICAI_PROXY_PORT` | `int` | `11435` | Live-Listen-Port für eingehende Client-Requests |
| `API_KEY` | `ACADEMICAI_PROXY_API_KEY` | `str (Secret)` | `"test-proxy-key-123456"` | Bearer Token für Client-Authentifizierung am Proxy |
| `BASE_URL` | `ACADEMICAI_BASE_URL` | `str` | `"https://academic-ai.boku.ac.at/api/v1"` | BOKU AcademicAI API-Basis-URL |
| `CLIENT_ID` | `ACADEMICAI_CLIENT_ID` | `str (Secret)` | `""` | BOKU Backend API Client ID |
| `CLIENT_SECRET` | `ACADEMICAI_CLIENT_SECRET` | `str (Secret)` | `""` | BOKU Backend API Client Secret |
| `HEALTH_CHECK_BACKEND` | `ACADEMICAI_HEALTH_CHECK_BACKEND` | `bool` | `True` | Aktiviert Backend-Connectivity-Check in `/health` |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | `ACADEMICAI_HEALTH_CHECK_TIMEOUT_SECONDS` | `float` | `2.0` | Timeout für Backend-Health-Check |
| `ENABLE_COST_MONITORING` | `ACADEMICAI_ENABLE_COST_MONITORING` | `bool` | `True` | Schaltet Cost-Header & Monitoring aktiv |
| `COST_CACHE_FILE` | `ACADEMICAI_COST_CACHE_FILE` | `str` | `"data/cost_cache.json"` | Pfad zur lokalen Cost-Cache-Datei |
| `COST_CACHE_TTL_SECONDS` | `ACADEMICAI_COST_CACHE_TTL_SECONDS` | `int` | `600` | Gültigkeitsdauer des Cost-Caches in Sekunden |
| `COST_REFRESH_TIMEOUT_SECONDS` | `ACADEMICAI_COST_REFRESH_TIMEOUT_SECONDS` | `float` | `8.0` | Timeout für Live-Refresh der Cost-API |
| `MAX_MESSAGES` | `ACADEMICAI_MAX_MESSAGES` | `int` | `200` | Maximal zulässige Anzahl an Chat-Nachrichten pro Request (Schutzgrenze 413) |
| `MAX_TOOLS` | `ACADEMICAI_MAX_TOOLS` | `int` | `64` | Maximal übermittelte Tool-Definitionen (Schutzgrenze 413) |
| `MAX_MESSAGE_TEXT_CHARS` | `ACADEMICAI_MAX_MESSAGE_TEXT_CHARS` | `int` | `500000` | Max. Zeichenlänge pro Einzelnachricht (Text/Prompt-Payload; Schutzgrenze 413) |
| `MAX_TOOL_SCHEMA_CHARS` | `ACADEMICAI_MAX_TOOL_SCHEMA_CHARS` | `int` | `100000` | Max. Zeichenlänge pro Tool-JSON-Schema (Schutzgrenze 413) |
| `MAX_REQUEST_JSON_CHARS` | `ACADEMICAI_MAX_REQUEST_JSON_CHARS` | `int` | `2000000` | Max. Gesamtgröße des Request-JSON-Strings (Schutzgrenze 413) |
| `RATE_LIMIT_PER_MINUTE` | `ACADEMICAI_RATE_LIMIT_PER_MINUTE` | `int` | `120` | Max. Anfragen pro Minute pro IP/Token-Bucket (Schutzgrenze 429) |
| `RATE_LIMIT_WINDOW_SECONDS` | `ACADEMICAI_RATE_LIMIT_WINDOW_SECONDS` | `int` | `60` | Zeitfenster für Rate Limiting in Sekunden |
| `ENABLE_HUMANIZATION_PASS` | `ACADEMICAI_ENABLE_HUMANIZATION_PASS` | `bool` | `False` | Aktiviert optionalen zweiten Pass zur Endantwort-Glättung |
| `HUMANIZATION_TEMPERATURE` | `ACADEMICAI_HUMANIZATION_TEMPERATURE` | `float` | `0.7` | LLM-Temperatur für Humanisierungspass |
| `DEFAULT_CHAT_TEMPERATURE` | `ACADEMICAI_DEFAULT_CHAT_TEMPERATURE` | `float` | `0.6` | Standard-Temperatur für Chat Completions |
| `DEFAULT_TOOL_TEMPERATURE` | `ACADEMICAI_DEFAULT_TOOL_TEMPERATURE` | `float` | `0.1` | Standard-Temperatur bei Requests mit Tools |
| `STREAM_CHUNK_DELAY_MS` | `ACADEMICAI_STREAM_CHUNK_DELAY_MS` | `int` | `0` | Künstliche Verzögerung zwischen SSE-Stream-Chunks (ms) |
| `DEBUG_DUMPS` | `ACADEMICAI_DEBUG_DUMPS` | `bool` | `False` | Schreibt Rohdaten nach `last_backend_request.json` |
| `ALLOWED_MODELS` | `ACADEMICAI_ALLOWED_MODELS` | `list[str]` | `["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini"]` | Modellkonstante für Verträge & Discovery |
| `PID_FILE` | `ACADEMICAI_PID_FILE` | `Path` | `Path("server.pid")` | Prozess-ID-Datei des laufenden Proxy-Daemons |
| `LOG_FILE_PATH` | `ACADEMICAI_PROXY_LOG_FILE` | `str` | `"server.log"` | Aktiver Log-Stream mit täglicher Rotation (30 Tage Retention) |
| `ERR_FILE_PATH` | `ACADEMICAI_PROXY_ERR_FILE` | `str` | `"server.err.log"` | Fehler-Log-Stream mit täglicher Rotation |
| `RETRY_MAX` | `ACADEMICAI_RETRY_MAX` | `int` | `2` | Maximale Retry-Wiederholungen bei Backend-Netzwerkfehlern |
| `RETRY_BASE_MS` | `ACADEMICAI_RETRY_BASE_MS` | `int` | `300` | Basis-Wartezeit für exponentielles Backoff bei Retries |

---

## 5. Request-Guards & Rate-Limiting-Zustand ([`academicai/request_guards.py`](../../academicai/request_guards.py))

Das Modul [`academicai/request_guards.py`](../../academicai/request_guards.py) kapselt die Inbound-Validierung von Chat-Completion-Payloads sowie das In-Memory Token-Bucket Rate-Limiting.

### Inbound-Validierung & Schutzgrenzen (413 / 422)
- **JSON-Serialisierbarkeit & Request-Größe (`validate_request_json_size`):**
  - Prüft, ob der Payload ein gültiges JSON-Objekt darstellt (422 bei Serialisierungsfehlern).
  - Vergleicht die aggregierte Zeichenlänge mit `MAX_REQUEST_JSON_CHARS` (413 bei Überschreitung).
- **Struktur- & Schema-Validierung (`validate_chat_request_body`):**
  - **422 Unprocessable Entity:** Nicht-Dict-Payloads, fehlendes oder leeres `model`, `model` > 200 Zeichen, fehlende/leere `messages`-Liste, nicht-objektbasierte Messages, fehlende/leere Message-Rollen, nicht-listenbasierte Tools/Functions, ungültige Tool-Objekte oder nicht serialisierbare Tools.
  - **413 Payload Too Large:** Nachrichtenanzahl > `MAX_MESSAGES`, extrahierter Plain-Text pro Nachricht > `MAX_MESSAGE_TEXT_CHARS`, Tool-Anzahl > `MAX_TOOLS`, serialisiertes Tool-Schema > `MAX_TOOL_SCHEMA_CHARS`.

### In-Memory Rate Limiting & Bucket Sweep TTL
- **Zustandsspeicher (`_rate_limit_buckets`):**
  - Dictionary `dict[str, list[float]]`, geschützt durch `_rate_limit_lock`.
  - Bucket-Schlüsselformat: `f"{client_host}:{token[:8]}"` via `_rate_limit_bucket`.
  - Hält Zeitstempel (`float`) erfolgreicher Anfragen innerhalb des aktiven Zeitfensters (`RATE_LIMIT_WINDOW_SECONDS`, Default: 60s).
- **Bucket Sweep / TTL Cleanup (`prune_rate_limit_buckets`):**
  - Zur Verhinderung unbegrenzten Speicherwachstums (Unbounded Memory Growth) bei vielen wechselnden Client-IPs/Tokens:
    1. **Automatischer periodischer Sweep:** In `_enforce_chat_rate_limit` wird nach Ablauf des Sweep-Intervalls (Default: 60s) ein automatischer Inline-Sweep ausgeführt.
    2. **On-Demand Pruning:** `prune_rate_limit_buckets(now=None, window_seconds=None)` kann jederzeit explizit aufgerufen werden.
    3. **Bereinigungsregel:** Buckets ohne aktive Zeitstempel (`all ts < now - window_seconds`) sowie leere Buckets werden vollständig gelöscht (`del _rate_limit_buckets[key]`). Verbleibende Buckets werden auf aktive Zeitstempel gekürzt.
- **Rückwärtskompatibilität:**
  - `server.py` re-exportiert `_validate_chat_request_body`, `_enforce_chat_rate_limit`, `_rate_limit_bucket`, `_rate_limit_buckets` und `_rate_limit_lock`.
  - Die interne Auflösung (`_get_limit`) prüft dynamisch Attribute auf dem `server`-Modul, sodass bestehende `monkeypatch.setattr(server, ...)`-Tests unverändert funktionieren.

