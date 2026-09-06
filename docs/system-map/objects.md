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
| `tool` | `user` | Standardisiertes Observation-Tag: `<tool_result id="<tool_call_id>" name="<tool_name>">\n<content>\n</tool_result>` (Attribute werden bei Fehlen weggelassen). |
| `assistant` (mit `tool_calls`) | `assistant` | Formatiert als `[Tool call: <name>(<arguments>)]`. |
| Aufeinanderfolgende gleiche Rollen | `user`/`assistant` | Werden zu einer einzigen Nachricht mit `\n\n` zusammengeführt. |

### `ToolCall` & High-Density TypeScript Signatures ([`academicai/tool_emulation.py`](../../academicai/tool_emulation.py))
Um das Prompt-Kontextfenster zu schonen, komprimiert `_compact_tool_def` OpenAI-JSON-Tools in prägnante TypeScript-Style-Signaturen:
- **Format:** `- name(param1: type, param2?: type = default) -- description`
- **Enums:** Werden als prägnante Typ-Unions dargestellt, z.B. `mode?: "exact" | "regex" | "fuzzy"`. Bei mehr als 4 Elementen erfolgt Truncation mit `...` (z.B. `"a" | "b" | "c" | "d" | ...`).
- **Getypte Arrays:** Inspektion von `items.type` (z.B. `string[]`, `number[]`, `boolean[]`, `object[]`) statt unspezifischem `any` oder `array`.
- **Defaults:** Explizite Angabe von Vorgabewerten (z.B. `limit?: number = 10`, `mode?: "exact" | "regex" = "exact"`).
- **Flache Objektkeys:** Bei `type: "object"` mit deklarierten `properties` erfolgt eine flache Key-Auflistung, z.B. `filter?: {query, tags}`.
- **Optionalität:** Pflichtparameter ohne `?`, optionale Parameter mit `?`.

### `tool_choice` Hard Enforcement & Prompt Constraints
- **`tool_choice: "required"`:** Injiziert zwingende Instruktionen, dass ein Tool-Call verpflichtend ist (`MANDATORY TOOL CALL`) und verbietet reine Textantworten (`{"action": "respond"}`).
- **Spezifisches Tool:** Bei Angabe eines Ziel-Tools (`{"type": "function", "function": {"name": "tool_x"}}` oder `"tool_x"`) wird die Modellantwort strikt auf diesen Tool-Namen festgelegt.
- **`tool_choice: "none"`:** Schließt Tool-Aufrufe explizit aus.

### JSON-Repair & Sanitization State
LLM-Antworten durchlaufen vor dem Parsing `_repair_and_load_json`:
- Entfernen von Trailing Commas vor `}` oder `]` (`,\s*([}\]])` $\to$ `\1`).
- Entschärfung nicht-escapeter Steuerzeichen / Zeilenumbrüche via `strict=False`.
- Reparatur fehlerhafter Backslash-Escapes (z.B. unvollständige Unicode-Sequenzen oder Windows-Pfade `C:\users\...`).
- Graceful Fallback auf sichere Standardwerte (`arguments = {}`) ohne Exception-Abstürze.

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

---

## 6. Cost-Monitoring & Cache-Lifecycle ([`academicai/cost_monitoring.py`](../../academicai/cost_monitoring.py))

Das Modul [`academicai/cost_monitoring.py`](../../academicai/cost_monitoring.py) kapselt die Kostenüberwachung des BOKU-Backends, das lokale Datei-Caching sowie die Generierung von Kosten-Headern.

### Lokales Caching & atomare Datei-Operationen
- **Atomares Schreiben (`write_cost_cache`):**
  - Schreibt neue Snapshots in eine temporäre Datei (`tempfile.NamedTemporaryFile`) im Zielverzeichnis, führt `flush()` und `os.fsync()` aus und ersetzt die Zieldatei atomar via `os.replace()`.
  - Windows-Absicherung: Enthält eine Retry-Schleife gegen kurzzeitige File-Sharing-Sperren (`PermissionError: [WinError 5]`).
  - Thread-Sicherheit: Schreib- und Lesezugriffe sind über ein reentrantes Thread-Lock (`_cost_lock = threading.RLock()`) geschützt.
  - Automatische Verzeichniserstellung (`p.parent.mkdir(parents=True, exist_ok=True)`).
- **Fehlertolerantes Lesen (`read_cost_cache`):**
  - Liefert bei fehlender oder korrupter Cache-Datei ein leeres Dictionary `{}` zurück, ohne Exceptions zu werfen.

### Stale-Erkennung & Lazy Background Refresh
- **Stale-Prüfung (`is_cost_cache_stale`):**
  - Vergleicht den UTC-Zeitstempel `updated_at` mit der aktuellen Zeit gegen `COST_CACHE_TTL_SECONDS` (Default: 600s).
  - Robust gegen unvollständige Payloads, ungültige Datumsformate und Zeitzonen-Mischungen.
- **Lazy Refresh (`get_cost_cache_with_lazy_refresh`):**
  - Liefert bei Chat-Completion-Requests sofort den vorhandenen Cache-Stand aus (non-blocking).
  - Erkennt einen abgelaufenen Cache (`is_cost_cache_stale`) und stößt bei Bedarf einen asynchronen Background-Task an (`refresh_cost_cache_background` via `loop.create_task` & `run_in_threadpool`).
  - Ein In-Flight-Guard (`_cost_refresh_in_flight`) verhindert parallele Stampede-Refreshes.

### Response-Header & Status-Payload
- **Header-Generierung (`build_cost_headers`):**
  - Liefert `{}` wenn `ENABLE_COST_MONITORING=False` oder der Cache leer ist.
  - Generiert bei aktivem Monitoring:
    - `X-AcademicAI-Cost-Stale`: `"true"` oder `"false"`
    - `X-AcademicAI-Cost-Updated-At`: ISO-8601-Zeitstempel
    - `X-AcademicAI-Total-Cost`: Formatierte Gesamtkosten (Fließkommazahl ohne überflüssige Nullen)
    - `X-AcademicAI-Total-Clients`: Anzahl erfasster Clients
    - `X-AcademicAI-Cost-Entries`: Anzahl an Einzelkosten-Einträgen
- **Status-Payload (`get_cost_status_payload`):**
  - Zentralisiert die Struktur für den internen Endpunkt `GET /internal/cost-status` mit Feldern `enabled`, `total_cost`, `total_clients`, `cost_entries`, `updated_at`, `is_stale`, `source`.

### Dynamische Attributauflösung & Rückwärtskompatibilität
- **Dynamische Konfiguration (`_get_setting`):**
  - Sucht Einstellungen (`ENABLE_COST_MONITORING`, `COST_CACHE_FILE`, `COST_CACHE_TTL_SECONDS`, `COST_REFRESH_TIMEOUT_SECONDS`) zuerst auf dem geladenen `server`-Modul und fällt danach auf `academicai.config` zurück.
  - Dadurch bleiben Unit- und Integrationstests, die Werte auf `server` mittels `monkeypatch.setattr(...)` überschreiben, uneingeschränkt funktionsfähig.
- **Server-Re-Exports:**
  - `server.py` re-exportiert alle wesentlichen Symbole (`_get_cost_cache_with_lazy_refresh`, `_build_cost_headers`, `_is_cost_cache_stale`, `_safe_float`, `_read_cost_cache`, `_write_cost_cache`, `_cost_lock`).

---

## 7. Runtime-Lifecycle & Health-Zustand ([`academicai/runtime.py`](../../academicai/runtime.py))

Das Modul [`academicai/runtime.py`](../../academicai/runtime.py) kapselt Lifecycle-Helfer für das Prozess- und Daemon-Management sowie die Zustandsermittlung des Proxies und dessen Upstream-Anbindung:

### PID-File Management (`write_pid_file`, `cleanup_pid_file`)
- **Prozessregistrierung (`write_pid_file`):**
  - Schreibt die PID des aktuellen Prozesses (`os.getpid()`) in `PID_FILE` (Default: `server.pid`).
  - Stellt sicher, dass das übergeordnete Verzeichnis existiert (`path.parent.mkdir(parents=True, exist_ok=True)`).
  - Fehlertolerant: Schlägt das Schreiben fehl (z.B. Dateisystem-Berechtigungen), wird eine Logging-Warnung erzeugt, ohne den Prozessstart abstürzen zu lassen.
- **Prozessabmeldung (`cleanup_pid_file`):**
  - Entfernt die PID-Datei nur, wenn die darin gespeicherte PID exakt der des aktuellen Prozesses entspricht (`raw == str(os.getpid())`).
  - Verhindert Race Conditions: Läuft bereits ein neuerer Prozess oder gehört die Datei einem anderen Prozess, bleibt sie unangetastet.
  - Ignoriert nicht existierende Dateien fehlertolerant.

### Backend-Connectivity Health Checks (`check_backend_health`)
- Führt bei aktiviertem Check (`HEALTH_CHECK_BACKEND=True`) einen synchronen HTTP-GET-Aufruf gegen den BOKU-Endpunkt `/api/v1/llm/models` via `httpx.Client` aus (Timeout konfiguriert über `HEALTH_CHECK_TIMEOUT_SECONDS`, Default: 2.0s).
- Misst die Latenz via `time.perf_counter()` in Millisekunden (`latency_ms`).
- Liefert ein strukturiertes Ergebnis-Dictionary zurück:
  - Bei Erfolg (HTTP 200): `{"enabled": True, "ok": True, "status_code": 200, "latency_ms": <int>}`
  - Bei Backend-Fehler (HTTP != 200): `{"enabled": True, "ok": False, "status_code": <code>, "latency_ms": <int>, "error": "backend responded with non-200 status"}`
  - Bei Exception (z.B. Timeout/Verbindungsabbruch): `{"enabled": True, "ok": False, "latency_ms": <int>, "error": "<str>"}`
  - Bei Deaktivierung (`HEALTH_CHECK_BACKEND=False`): `{"enabled": False, "ok": None}`

### Standardisierte Health-Payload-Generierung (`get_health_payload`)
- Erstellt das finale JSON-Payload für den öffentlichen `/health`-Endpunkt:
  - `status`: `"ok"` (wenn Backend erreichbar oder Check deaktiviert), `"degraded"` (wenn Check aktiviert und Backend nicht erreichbar).
  - `service`: `"academicai-proxy"`
  - `backend`: Enthaltenes Dictionary aus `check_backend_health()`.
- Automatischer Fallback: Wird `get_health_payload()` ohne Backend-Parameter aufgerufen, führt es selbstständig `check_backend_health()` aus.

### Dynamische Attributauflösung & Server-Kompatibilität
- **Dynamische Konfiguration (`_get_setting`):**
  - Prüft Attribute auf dem geladenen `server`-Modul (`PID_FILE`, `HEALTH_CHECK_BACKEND`, `HEALTH_CHECK_TIMEOUT_SECONDS`, `get_base_url`, `get_headers`) vor dem Fallback auf `academicai.config`.
  - Dadurch bleiben Test-Suites mit `monkeypatch.setattr(server, ...)` vollständig abwärtskompatibel.
- **Server-Re-Exports:**
  - `server.py` re-exportiert `write_pid_file`, `_write_pid_file`, `cleanup_pid_file`, `_cleanup_pid_file`, `check_backend_health`, `_check_backend_health`, `get_health_payload`.

---

## 8. Logging-Infrastruktur & Handler-State ([`academicai/logging_config.py`](../../academicai/logging_config.py))

Das Modul [`academicai/logging_config.py`](../../academicai/logging_config.py) kapselt die Initialisierung, Rotation und Entkopplung des Logging-Subsystems:

### Handler- und Logger-Hierarchie
- **Formatter (`log_formatter`):**
  - Standardisiertes Zeit- und Level-Format: `%(asctime)s %(levelname)s %(message)s`.
- **Info File Handler (`info_handler`):**
  - `TimedRotatingFileHandler` auf Pfad `LOG_FILE_PATH` (Default: `server.log`, tägliche Rotation `when="D"`, `interval=1`, 30 Tage Retention, `encoding="utf-8"`).
  - Minimum Level: `level` (Default: `logging.INFO`).
- **Error File Handler (`error_handler`):**
  - `TimedRotatingFileHandler` auf Pfad `ERR_FILE_PATH` (Default: `server.err.log`, tägliche Rotation `when="D"`, `interval=1`, 30 Tage Retention, `encoding="utf-8"`).
  - Minimum Level: `logging.ERROR`.
- **Console Handler (`console_handler`):**
  - `logging.StreamHandler(sys.stdout)` für direkte Terminal- und Container-Ausgabe mit Level `level` (Default: `logging.INFO`).
- **Root Logger (`root_logger`):**
  - Befestigt `info_handler`, `error_handler` und `console_handler` am Root-Logger von Python.
- **Proxy Logger (`log` / `get_logger`):**
  - Dedizierter Named-Logger `academicai-proxy` für anwendungsspezifische Traces und strukturierte Warnungen/Fehler.

### Uvicorn-Logger-Wiring
- Konfiguriert die Uvicorn-Server-Logger (`uvicorn`, `uvicorn.error`, `uvicorn.access`):
  - Bereinigt Standard-Handler (`ulog.handlers = []`) und leitet alle Uvicorn-Logs direkt an `info_handler`, `error_handler` und `console_handler`.
  - Setzt `ulog.propagate = False`, um doppelte Protokollierungen über den Root-Logger zu unterbinden.

### Windows-kompatible Bereinigung & dynamische Auflösung
- **Ressourcenbereinigung (`close_handlers`):**
  - Schließt aktive File-Handler und dereferenziert sie aus Root- und Uvicorn-Loggern, um gesperrte Dateihandles unter Windows bei Rekonfigurationen oder Test-Teardowns zu verhindern.
- **Dynamische Attributauflösung (`_get_setting`):**
  - Löst `LOG_FILE_PATH` und `ERR_FILE_PATH` dynamisch über Parameter, Attribute auf `server` oder `academicai.config` auf.
- **Server-Rückwärtskompatibilität:**
  - `server.py` re-exportiert `log`, `log_formatter`, `info_handler`, `error_handler`, `console_handler`, `root_logger`, `configure_logging`, `get_logger`, `close_handlers`, `log_file_path`, `err_file_path`.


