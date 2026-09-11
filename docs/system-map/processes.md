# System Map — Processes (Verben, Workflows & Pipelines)

> **Komponente**: Kontrollflüsse, Pipelines und dynamische Transformationen  
> **Kontext**: [`README.md`](README.md)

---

## 1. Server Startup & Request Lifecycle (`academicai/app.py`, `server.py`)

### Server Startup & Lifespan Pipeline
Der Server-Lifecycle wird modern über FastAPI Lifespan Events gesteuert:

```
[Server Process Launch (server.py / uvicorn)]
       │
       ▼
 1. Lifespan Startup (academicai/app.py: lifespan):
    ├─ validate_config() (Proxy-API-Key Integritätsprüfung)
    └─ write_pid_file() (Registrierung der aktuellen PID in server.pid)
       │
       ▼
  2. Request Handling Phase (Active HTTP Server):
     ├─ GET /health (Backend Health Check & Latenzmessung)
     ├─ GET /internal/cost-status (Cost Status Snapshot aus Cache)
     ├─ GET /v1/models (Modell-Discovery)
     ├─ POST /v1/chat/completions (OpenAI Chat Completions via _execute_completion_pipeline)
     └─ POST /v1/responses (OpenAI Responses API für Codex CLI & Desktop via _execute_completion_pipeline)
        │
        ▼
  3. Lifespan Shutdown (academicai/app.py: lifespan finally):
     └─ cleanup_pid_file() (Sicheres Entfernen der PID-Datei bei passendem PID-Match)
```

### Shared Execution Engine: `_execute_completion_pipeline`
Sowohl `POST /v1/chat/completions` als auch `POST /v1/responses` nutzen intern dieselbe zentrale Ausführungspipeline in [`academicai/app.py`](../../academicai/app.py):
1. **Normalisierung:** `/v1/responses` normalisiert `instructions`, `input` und Tools via `academicai.responses.normalize_responses_request` in das kanonische Format (`messages`, `tools`).
2. **Ausführung:** `_execute_completion_pipeline` führt Message-Normalisierung, Sticky-System-Prepending, Tool-Injektion, LLM-Backend-Call, Tool-Call-Parsing und optionale Humanisierung durch und liefert ein typisiertes `CanonicalResult`.
3. **Formatierung:**
   - Chat Completions: Formatiert als Standard-OpenAI-Response (`build_tool_calls_response` oder `build_tool_calls_sse_chunks`).
   - Responses API: Formatiert als OpenAI-Responses-Objekt (`build_responses_output`) oder Responses-SSE-Events (`build_responses_sse_events`).

### Request Lifecycle: `POST /v1/chat/completions`

Jeder Chat-Completion-Request durchläuft eine 8-Stufen-Pipeline in [`academicai/app.py`](../../academicai/app.py) (exponiert über [`server.py`](../../server.py)), gesteuert durch die zentralen Richtlinien und Limits aus [`academicai/config.py`](../../academicai/config.py):

```
[Inbound Client Request]
       │
       ▼
 1. Authentifizierung & Insecure-Key-Check (Bearer Token via API_KEY aus academicai/config.py)
       │
       ▼
 2. Payload-Validierung & Schutzgrenzen (academicai/request_guards.py & Rate-Limiting):
    ├─ JSON-Größe vs. ACADEMICAI_MAX_REQUEST_JSON_CHARS via validate_request_json_size (413 bei Übergröße)
    ├─ Nachrichtenanzahl & Textlänge vs. ACADEMICAI_MAX_MESSAGES / MAX_MESSAGE_TEXT_CHARS via validate_chat_request_body (413)
    ├─ Tools-Anzahl & Schema-Größe vs. ACADEMICAI_MAX_TOOLS / MAX_TOOL_SCHEMA_CHARS (413)
    ├─ Strukturiertes Logging in server.log bei jeder Abweisung (413/422/400)
    └─ Token-Bucket Rate-Limiting mit periodischem TTL-Sweep (academicai/request_guards.py) vs. RATE_LIMIT_PER_MINUTE / RATE_LIMIT_WINDOW_SECONDS (429)
       │
       ▼
 3. Message-Normalisierung & Heuristiken (academicai/transformation.py & humanization.py & tool_emulation.py):
    ├─ last_user_text & extract_text_content (Plain-Text-Extraktion via academicai/transformation.py)
    ├─ is_human_readable_target (Human Channel vs. Cron)
    └─ apply_post_tool_guard (Fehler-Schutz nach Tool-Result in academicai/tool_emulation.py)
       │
       ▼
 4. Tool-Injektion (inject_tools_into_messages in academicai/tool_emulation.py)
    └─ Wandelt JSON-Tools in System-Prompt-Instruktionen um
       │
       ▼
 5. AcademicAI-Backend HTTP-Aufruf (academicai.provider)
    └─ Übertragung mit X-Client-ID / X-Client-Secret & Azure Prefix Cache
       │
       ▼
 6. Parsing:
    └─ parse_tool_calls (Extraktion von ```json ... ``` Calls)
       │
       ▼
 7. Humanisierung & Response-Glättung (academicai/humanization.py):
    └─ Optionaler 2. LLM-Pass (run_humanization_pass / build_humanization_messages) bei ENABLE_HUMANIZATION_PASS auf Human-Kanälen
       │
       ▼
 8. Response-Formatierung:
    ├─ Streaming: SSE-Chunk-Generator (build_tool_calls_sse_chunks)
    └─ Non-Streaming: JSON Response (build_tool_calls_response)
       │
       ▼
[Outbound Client Response]
```

---

## 2. Tool-Emulation Pipeline ([`academicai/tool_emulation.py`](../../academicai/tool_emulation.py))

> Detailliertes Architektur- und Entwurfskonzept: [`../architecture/concept-tool-emulation.md`](../architecture/concept-tool-emulation.md)  
> *Hinweis:* Die Kernheuristiken der Tool-Emulation wurden während der Modularisierung nicht verhaltensändernd modifiziert.

Da das AcademicAI-Backend native Tool-Calling-Felder ignoriert, nutzt der Proxy eine synthetische Emulation:

1. **High-Density TypeScript Signatures (`_compact_tool_def`):**  
   Komprimiert Tool-Definitionen in einzeilige TypeScript-Signaturen mit Enums als Unions, getypten Arrays (`string[]`, `number[]`), Defaults und flachen Objekten (`filter?: {query, tags}`).
2. **Prompt-Injektion & `tool_choice` Hard Enforcement (`inject_tools_into_messages`):**  
   Injiziert Tools in den System-Prompt. Unterstützt harte Vorgaben (`tool_choice: "required"`, spezifische Tools oder `"none"`), die alternative Antworten verbieten. Ergänzend wird eine JSON-Erinnerungsnachricht ans Nachrichtenende angehängt.
3. **Standardisierte Observation Tags (`academicai/transformation.py`):**  
   Vorherige Tool-Ergebnisse werden als strukturierte `<tool_result id="{id}" name="{name}">\n{content}\n</tool_result>` Blöcke in die Benutzerhistorie überführt.
4. **JSON-Repair & Extraktion (`parse_tool_calls`, `_repair_and_load_json`):**  
   Entfernt Trailing Commas, repariert unescapte Steuerzeichen/Newlines via `strict=False`, korrigiert Backslash-Escapes und isoliert `tool_call` bzw. `tool_calls` Payloads.
5. **Fallback-Handling:**  
   Falls das Modell JSON ausgibt, obwohl ein Mensch im Chat sitzt ([`is_human_readable_target`](../../academicai/humanization.py)), formatiert [`format_arbitrary_json_for_humans`](../../academicai/tool_emulation.py) das JSON in lesbaren Fließtext um.

---

## 3. Post-Tool Result Guard (`_apply_post_tool_guard`)

Verhindert Endlos-Schleifen oder falsche Erfolgsmeldungen:
- **Fehler im Tool-Result (`error:`, `cannot parse`, `failed`):**  
  Injiziert eine System-Warning: *"TOOL_RESULT_ERROR: The latest tool result contains an error. Do NOT claim success. Issue a corrected tool call or report the failure."*
- **Erfolg im Tool-Result:**  
  Injiziert: *"NO_FURTHER_TOOL_CALLS: Produce the final user-facing answer."*

---

## 4. Resilience & Retry-Schleife ([`academicai/provider.py`](../../academicai/provider.py))

HTTP-Aufrufe an das AcademicAI-Backend sind gegen transiente Netzwerkfehler abgesichert:
- **Retry-Limit:** `ACADEMICAI_RETRY_MAX` (Default: 2 Wiederholungen).
- **Backoff:** Exponentielles Backoff (`RETRY_BASE_MS * 2^attempt`) bei HTTP 502/503/504 oder `httpx.TransportError`.
- **Fast-Fail:** HTTP 401/403 bricht sofort ab (kein Retry bei Auth-Fehlern).

---

## 5. Cost-Monitoring & Lazy Cache Lifecycle ([`academicai/cost_monitoring.py`](../../academicai/cost_monitoring.py))

Das Modul überwacht Kosten und Quoten des AcademicAI-Backends non-blocking:
1. **Cache Read & Stale Evaluation (`get_cost_cache_with_lazy_refresh`, `is_cost_cache_stale`):**
   - Eingehende Requests lesen den bestehenden Cache via `read_cost_cache`.
   - Ist der Cache älter als `COST_CACHE_TTL_SECONDS` (Default: 600s), wird ein asynchroner Refresh im Hintergrund angestoßen (`refresh_cost_cache_background`).
2. **Asynchroner Live-Snapshot (`fetch_cost_snapshot`):**
   - Ruft `GET /api/v1/cost/` am AcademicAI-Backend mit konfigurierten Credentials ab.
   - Extrahiert `total_cost`, `total_clients` und `cost_entries` via `_extract_cost_summary`.
3. **Atomares Schreiben (`write_cost_cache`):**
   - Schreibt den neuen Cache atomar (`tempfile` + `os.replace` mit Windows-Retry und `_cost_lock`), um Race Conditions zwischen parallelen Requests zu verhindern.
4. **Header-Generierung (`build_cost_headers`):**
   - Injiziert `X-AcademicAI-Total-Cost`, `X-AcademicAI-Total-Clients`, `X-AcademicAI-Cost-Entries`, `X-AcademicAI-Cost-Updated-At` und `X-AcademicAI-Cost-Stale` in ausgehende Chat-Responses.

---

## 6. Runtime-Lifecycle & Health-Check-Pipeline ([`academicai/runtime.py`](../../academicai/runtime.py))

Verwaltet den Server-Daemon-Lifecycle und die Überwachung der Upstream-Verbindung:

1. **Server-Startup (`_on_startup`):**
   - Ruft `validate_config()` zur Integritätsprüfung des Proxy-API-Keys auf.
   - Ruft `_write_pid_file()` auf, stellt sicher, dass das Zielverzeichnis existiert, und schreibt die aktuelle Prozess-PID atomar in `PID_FILE`.
2. **Server-Shutdown (`_on_shutdown`):**
   - Ruft `_cleanup_pid_file()` auf. Prüft, ob die Datei existiert und deren Inhalt exakt der aktuellen PID entspricht, bevor sie gelöscht wird (verhindert Löschung fremder/neuerer Daemon-Dateien).
3. **Health-Check-Workflow (`GET /health`):**
   - Ruft `_check_backend_health()` auf:
     - Prüft `HEALTH_CHECK_BACKEND`. Falls deaktiviert, Rückgabe von `{"enabled": False, "ok": None}`.
     - Falls aktiviert: HTTP-GET auf AcademicAI `/api/v1/llm/models` mit Timeout `HEALTH_CHECK_TIMEOUT_SECONDS` und Auth-Headern.
     - Ermittelt Antwortstatus und misst Request-Latenz via `time.perf_counter()`.
   - Übergibt das Ergebnis an `get_health_payload(backend)`:
     - Berechnet Gesamtstatus (`"ok"` bzw. `"degraded"` bei `enabled=True` und `ok=False`).
     - Liefert standardisiertes JSON-Payload für Monitoring und Health-Probes.

---

## 7. Logging-Initialisierung & Uvicorn-Wiring ([`academicai/logging_config.py`](../../academicai/logging_config.py))

Das Logging-Subsystem wird zentral über `configure_logging()` orchestriert:

1. **Dynamische Pfadauflösung & Vorbereitung:**
   - Ermittelt `LOG_FILE_PATH` und `ERR_FILE_PATH` über explizite Argumente, Attribute auf dem `server`-Modul oder `academicai.config`.
   - Stellt sicher, dass Zielverzeichnisse für Log-Dateien existieren (`Path.mkdir(parents=True, exist_ok=True)`).
2. **Sauberes Teardown existierender Handler (`close_handlers`):**
   - Schließt offene File-Handler und dereferenziert sie vom Root-Logger sowie allen Uvicorn-Loggern, um gesperrte Dateihandles unter Windows zu vermeiden.
3. **Handler-Initialisierung & Root-Logger-Konfiguration:**
   - Instanziiert `TimedRotatingFileHandler` für Standard-Logs (täglich, 30 Tage Retention, UTF-8) und Error-Logs (täglich, 30 Tage Retention, Level `ERROR`, UTF-8).
   - Instanziiert `StreamHandler(sys.stdout)` für Konsolen-Ausgaben.
   - Formatiert alle Handler mit `%(asctime)s %(levelname)s %(message)s`.
   - Registriert alle drei Handler am Python Root-Logger (`logging.getLogger()`).
4. **Uvicorn-Logger-Wiring:**
   - Konfiguriert die Logger `uvicorn`, `uvicorn.error` und `uvicorn.access`.
   - Setzt `propagate = False` und weist ihnen die gemeinsamen rotierenden File- und Konsolen-Handler zu.
5. **Server-Synchronisation:**
   - Spiegelt die aktiven Handler- und Logger-Instanzen (`info_handler`, `error_handler`, `console_handler`, `root_logger`, `log`) auf das `server`-Modul für vollständige Rückwärtskompatibilität.

---

## 8. Humanization Flow & Zweiter Pass ([`academicai/humanization.py`](../../academicai/humanization.py))

Verwandelt strukturierte Tool-Ausgaben für menschliche Chat-Kanäle in natürliche Konversationstexte:

1. **Zielkanal-Klassifizierung (`is_human_readable_target`):**
   - Prüft System-Prompts auf bekannte Messenger- und Chat-Tags (`channel=whatsapp`, `telegram`, `signal`, `discord`, `slack` etc.) sowie OpenClaw-Metadaten (`conversation info`, `is_group_chat`, `sender`).
   - Maschinen-Override: Enthält die Benutzeranfrage ein Cron-Präfix (`[cron:`), wird Humanisierung strikt unterdrückt (`False`).
2. **User-Query-Extraktion (`last_user_text`):**
   - Extrahiert die letzte Benutzeranfrage aus der Historie (unterstützt Plain-String und strukturierte Content-Parts via `extract_text_content`).
3. **Zweiter LLM-Pass (`run_humanization_pass`, `build_humanization_messages`):**
   - Bei aktiver Humanisierung (`ENABLE_HUMANIZATION_PASS=True`) und erkanntem Human-Kanal:
   - Erstellt einen dedizierten Prompt (`build_humanization_messages`), der das Modell anweist, strukturierte JSON-/Tool-Ergebnisse in eine prägnante, natürliche Antwort ohne Metadaten oder Codeblöcke umzuschreiben.
   - Führt den Request asynchron über `run_in_threadpool(completion, ...)` aus (unterstützt synchrone und asynchrone Callables).
   - Bei Fehlern oder leerer Rückgabe erfolgt ein fehlertoleranter Fallback auf die First-Pass-Antwort.

---

## 9. Test- & Regressionsarchitektur ([`tests/`](../../tests/))

Die Test-Suiten decken die sensiblen Transformations- und Sicherheitsheuristiken ab und sichern die Schnittstellenverträge vor Refactorings:

| Test-Suite | Testfokus & Schutzbereich |
| :--- | :--- |
| `test_characterization_endpoints.py` | Charakterisierungssuite für öffentliche Schnittstellenverträge (`GET /health`, `GET /internal/cost-status`, `GET /v1/models`, `POST /v1/chat/completions`) mittels `fastapi.testclient.TestClient`. |
| `test_tool_emulation.py` | E2E- & Kern-Tests für JSON-Mode Responses und Fallback-Formatierer. |
| `test_tool_emulation_unit.py` | Unit-Tests für TypeScript-Style Tool-Signaturen, hard `tool_choice` Enforcement, JSON-Repair Sanitization (Trailing Commas, Escapes) und `<tool_result>` XML-Tags. |
| `test_multi_step_tool_emulation.py` | Mehrstufige Handoffs: Tool Call → Result → Next Call → Final Answer. |
| `test_post_tool_guard.py` | Verhindert Endlosschleifen nach Tool-Fehlern oder phantomhaften Folgeaufrufen. |
| `test_humanization_flow.py` | Unit- & Integrationstests für `academicai/humanization.py`: Zielkanal-Erkennung (`is_human_readable_target`: Messenger/OpenClaw vs. Cron-Override), User-Text-Extraktion (`last_user_text`: String, Multi-Part, Fallbacks), Prompt-Konstruktion (`build_humanization_messages`), Pass-Ausführung (`run_humanization_pass`: Erfolg, Fehler-Fallback, Empty-Content, Async-Callables) sowie Chat-Endpoint-Integration (`ENABLE_HUMANIZATION_PASS` Toggle). |
| `test_hardening_security_runtime.py` | Schutz gegen Klartext-Leakage, Insecure Key Detection, Request-Guards & Payload-Limits. |
| `test_transformation_sticky_system.py` | Korrektes Prependen von System-Prompts an erste User-Message (Azure Prefix Caching). |
| `test_config.py` | Validiert Standardwerte, Env-Override, sicheren Import ohne fatalen Crash, Insecure-Key-Validierung und Rückwärtskompatibilität. |
| `test_request_guards.py` | Validiert Inbound-Payloads (422/413), JSON-Größenlimits, Token-Bucket Rate-Limiting (429), Bucket-Sweep / TTL-Cleanup gegen unbegrenztes Speicherwachstum sowie Server-Re-Exports. |
| `test_cost_monitoring_unit.py` | Unit-Tests für akademische Kostenüberwachung: Parsing (`_safe_float`, `_parse_iso_ts`), Payload-Extraktion (`_extract_cost_summary`), Stale-Prüfung (`is_cost_cache_stale`), Header-Generierung (`build_cost_headers`), atomare Cache-Roundtrips, Thread-Sicherheit und Server-Re-Exports. |
| `test_runtime_unit.py` | Unit-Tests für Laufzeit-Lifecycle: PID-File-Erstellung und -Bereinigung mit PID-Matching, Backend-Health-Checks (Erfolg, Timeout, Fehler, Deaktivierung), Health-Payload-Generierung (`ok`/`degraded`) und Server-Re-Exports. |
| `test_logging_config_unit.py` | Unit-Tests für Logging-Konfiguration: Formatter, rotierende File-Handler (Info & Error), Konsolen-Handler, Uvicorn-Logger-Wiring (propagate=False), dynamische Pfadauflösung, Windows-kompatibles Schließen via close_handlers und Server-Re-Exports. |
| `test_app_unit.py` | Unit-Tests für Application Factory (`create_app`), modernen ASGI-Lifespan (Startup/Shutdown Hooks, Starlette TestClient), Routing-Delegation (`/health`, `/internal/cost-status`, `/v1/models`, `/v1/chat/completions`), Fehlerbehandlung (502) sowie `apply_post_tool_guard` und Server-Re-Exports. |
| `test_responses_api.py` | Umfassende Suite für OpenAI Responses API (`POST /v1/responses`): Normalisierung (Top-level instructions, strukturierte input-Items, flat/nested tools), Validierung (422/413), Non-Streaming Serialization, vollständige SSE-Event-Sequenz, Rust-kompatibles Token-Accounting (`input_tokens`/`output_tokens`), Authentifizierung und Multi-Turn-Tool-Roundtrip mit sandbox-basiertem `function_call_output`. |
| `run_local_tests.ps1` | Lokaler Test-Runner: Führt Offline-Tests aus bzw. startet im E2E-Modus den isolierten Test-Server auf **Port 11436**, führt `pytest` aus und stoppt den Server sauber via PID. |


### Sicherheits-Baselines der Testumgebung
- **Test-Discovery-Scope (`pytest.ini`):** Über `testpaths = tests` wird Pytest angewiesen, Tests ausschließlich im Verzeichnis `tests/` zu suchen. Dadurch werden Diagnose- und Connectivity-Skripte im Root-Verzeichnis (wie `test_models_connectivity.py`) von der Testausführung ausgeschlossen.
- **Test-Port-Isolation (`tests/_local_env.py`):** Als Fallback für `ACADEMICAI_TEST_BASE_URL` ist Port `11436` (`http://127.0.0.1:11436`) vorkonfiguriert. Dies verhindert versehentliche Netzwerkaufrufe gegen eine parallel laufende produktive Instanz auf Port `11435`, falls Umgebungsvariablen nicht explizit gesetzt sind.



