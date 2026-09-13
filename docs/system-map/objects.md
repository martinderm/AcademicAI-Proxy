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

### `POST /v1/responses` (OpenAI Responses API für Codex CLI & Desktop)
Responses-API-Payload von OpenAI Codex:
- **`model`** `(str, required)`: Zielmodell (z. B. `gpt-4o`, `gpt-4o-mini`).
- **`instructions`** `(str, optional)`: System-Prompt auf oberster Ebene (wird zu `role: system` normalisiert).
- **`input`** `(list[dict] | str, required)`: Liste strukturierter Eingabeelemente oder Freitext:
  - `type: "message"`: Rolle (`developer`, `system`, `user`, `assistant`), `content` (`str` oder Array von Text-Parts wie `input_text`).
  - `type: "function_call"`: Tool-Aufruf von Codex (`call_id`, `name`, `arguments`).
  - `type: "function_call_output"`: Tool-Ergebnis nach lokaler Ausführung in Codex (`call_id`, `output`).
- **`tools`** `(list[dict], optional)`: Tool-Definitionen (unterstützt sowohl flaches Format mit `name` auf Root-Ebene als auch verschachteltes `function`-Format).
- **`stream`** `(bool, default: False)`: SSE-Streaming aktivieren (`text/event-stream`).
- **`temperature`**, **`max_output_tokens`** `(optional)`: Generierungsparameter.

### `GET /v1/models`
Liefert verfügbare Modelle als OpenAI ModelList (direkt aus dem lokalen 24h `ModelCatalog` ohne Upstream-Netzwerklatenz):
```json
{
  "object": "list",
  "data": [
    {
      "id": "gpt-5-mini",
      "object": "model",
      "created": 0,
      "owned_by": "academicai",
      "context_window": 400000,
      "max_tokens": 128000,
      "costs": [
        { "cost": 0.00028, "costType": "input_tokens" },
        { "cost": 0.0022, "costType": "output_tokens" }
      ]
    }
  ]
}
```

---

## 2. Outbound Backend Schemas (Proxy $\to$ AcademicAI)

Das AcademicAI-Backend verlangt spezifische Auth-Header und REST-Strukturen ([`academicai/auth.py`](../../academicai/auth.py)):
- **Headers**:
  - `X-Client-ID`: Client-ID aus Umgebungsvariable `ACADEMICAI_CLIENT_ID`.
  - `X-Client-Secret`: Secret aus `ACADEMICAI_CLIENT_SECRET`.
  - `Content-Type`: `application/json`.
- **Payload**:
  - Zusammengeführte Prompt-Struktur, optimiert für das Azure-Prefix-Caching (System-Instruktionen am Kopf der ersten User-Message).

---

## 3. Emulations- & Transformationsobjekte

### Message-Rollen-Normalisierung ([`academicai/transformation.py`](../../academicai/transformation.py))
Das AcademicAI-Backend unterstützt ausschließlich die Rollen **`user`** und **`assistant`**. Eingehende OpenAI-Rollen werden daher deterministisch transformiert:

| Eingehende OpenAI-Rolle | Zielrolle am AcademicAI-Backend | Transformationsregel |
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
| `BASE_URL` | `ACADEMICAI_BASE_URL` | `str` | `"https://academic-ai.boku.ac.at/api/v1"` | AcademicAI API-Basis-URL |
| `CLIENT_ID` | `ACADEMICAI_CLIENT_ID` | `str (Secret)` | `""` | AcademicAI Backend API Client ID |
| `CLIENT_SECRET` | `ACADEMICAI_CLIENT_SECRET` | `str (Secret)` | `""` | AcademicAI Backend API Client Secret |
| `HEALTH_CHECK_BACKEND` | `ACADEMICAI_HEALTH_CHECK_BACKEND` | `bool` | `True` | Aktiviert Backend-Connectivity-Check in `/health` |
| `HEALTH_CHECK_TIMEOUT_SECONDS` | `ACADEMICAI_HEALTH_CHECK_TIMEOUT_SECONDS` | `float` | `2.0` | Timeout für Backend-Health-Check |
| `ENABLE_COST_MONITORING` | `ACADEMICAI_ENABLE_COST_MONITORING` | `bool` | `True` | Schaltet Backend Cost-Header & Monitoring aktiv |
| `COST_CACHE_FILE` | `ACADEMICAI_COST_CACHE_FILE` | `str` | `"data/cost_cache.json"` | Pfad zur lokalen Cost-Cache-Datei |
| `COST_CACHE_TTL_SECONDS` | `ACADEMICAI_COST_CACHE_TTL_SECONDS` | `int` | `600` | Gültigkeitsdauer des Cost-Caches in Sekunden |
| `COST_REFRESH_TIMEOUT_SECONDS` | `ACADEMICAI_COST_REFRESH_TIMEOUT_SECONDS` | `float` | `8.0` | Timeout für Live-Refresh der Cost-API |
| `ENABLE_LOCAL_COST_TRACKING` | `ACADEMICAI_ENABLE_LOCAL_COST_TRACKING` | `bool` | `True` | Schaltet autonome lokale Kostenberechnung & Header aktiv |
| `MODEL_CATALOG_FILE` | `ACADEMICAI_MODEL_CATALOG_FILE` | `str` | `"data/model_catalog.json"` | Pfad zum persistenten 24h Modellkatalog & Preistabelle |
| `MODEL_CATALOG_TTL_SECONDS` | `ACADEMICAI_MODEL_CATALOG_TTL_SECONDS` | `int` | `86400` | Gültigkeitsdauer des Modellkatalogs in Sekunden (24h) |
| `LOCAL_COST_CACHE_FILE` | `ACADEMICAI_LOCAL_COST_CACHE_FILE` | `str` | `"data/local_cost_cache.json"` | Pfad zur persistenten Aggregationsdatei |
| `LOCAL_COST_HISTORY_LIMIT` | `ACADEMICAI_LOCAL_COST_HISTORY_LIMIT` | `int` | `500` | Maximale Einträge im Ringpuffer der Request-Historie |
| `COST_CURRENCY` | `ACADEMICAI_COST_CURRENCY` | `str` | `"EUR"` | Währung für lokale Abrechnung & Response-Header |
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

Das Modul [`academicai/cost_monitoring.py`](../../academicai/cost_monitoring.py) kapselt die Kostenüberwachung des AcademicAI-Backends, das lokale Datei-Caching sowie die Generierung von Kosten-Headern.

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

## 7. Lokale Kostenberechnung & Persistenter Aggregator ([`academicai/cost_calculation.py`](../../academicai/cost_calculation.py), [`academicai/local_cost_tracker.py`](../../academicai/local_cost_tracker.py))

Kapselt die vollkommen autonome, anfragegenaue Kostenermittlung ohne Abhängigkeit vom geschützten AcademicAI-Endpunkt `/api/v1/cost/`:

### Lokaler 24h Modellkatalog & Preistabelle (`ModelCatalog`, `ModelEntry`)
- **Einheiten-Normalisierung (`parse_model_costs`):**
  - AcademicAI liefert Preise im Feld `costs` pro **1.000 Tokens (1k Tokens)**.
  - Normalisierte Rate pro Einzeltoken: `Decimal(cost) / Decimal(1000)`.
  - Bei gestaffelten Preisen (`costs` mit mehreren `input_tokens`/`output_tokens`-Einträgen wie `gpt-5.5`, `gemini-2.5-pro`) wird garantiert die **niedrigste Preisstufe** als Baseline (`input_cost_per_token`, `output_cost_per_token`) gewählt und `is_tiered = True` gesetzt.
  - **Strukturierte, sortierte Tiers (`tiers`):** Sämtliche Tarifstufen werden aufsteigend sortiert im Modellobjekt gespeichert:
    - **Tier 1 (`short_context`):** $\le 128\text{k}$ Prompt-Tokens (`max_prompt_tokens: 128000`) mit dem Basis-Tarif.
    - **Tier 2 (`long_context`):** $> 128\text{k}$ Prompt-Tokens (`max_prompt_tokens: context_window`) mit den erhöhten Upstream-Raten.
    - Entspricht den Upstream-Tarifmodellen von Google Cloud Vertex AI (Gemini Pro) und Microsoft Azure OpenAI (GPT-5.5 ShortCo vs. LongCo).
  - Parst und serialisiert zudem Modellmetadaten wie `context_window` (`contextWindow`) und `output_token_limit` (`outputTokenLimit`).
- **Thread- und Async-sicherer Katalog (`ModelCatalog`):**
  - Gesteuert über `MODEL_CATALOG_TTL_SECONDS` (Default: 86400s / 24 Stunden) und `MODEL_CATALOG_FILE` (Default: `"data/model_catalog.json"`).
  - **Atomare JSON-Dateipersistenz:** Der Modellkatalog wird auf Platte gespeichert und beim Serverstart sofort ohne Netzwerklatenz geladen.
  - **Öffentlicher Endpunkt (`GET /v1/models`):** Liefert direkt aus dem lokalen In-Memory-Katalog im Standard-OpenAI-Format (`to_openai_models_response`) in < 1ms Antwortzeit ohne blockierenden Upstream-Roundtrip.
  - Lazy Background Refresh (`trigger_background_refresh`) ohne Request-Blockade nach Ablauf der 24h-TTL.
  - Fehlertoleranter Fallback: Bleibt bei Ausfall der AcademicAI-Upstream-API transparent auf dem zuletzt gespeicherten Stand.

### Hochpräzise Request-Kostenberechnung (`RequestCost`, `calculate_request_cost`)
- **Decimal-Arithmetik:**
  - Berechnet `input_cost`, `output_cost`, `per_request_cost` und `request_cost` mit Pythons `Decimal`, um Fließkomma-Drift bei Mikro-Beträgen zu eliminieren.
- **Standardisierte Response-Header (`to_headers`):**
  - `X-AcademicAI-Request-Cost`: Formatierter Betrag (z.B. `"0.000825"`).
  - `X-AcademicAI-Input-Cost`: Berechnete Prompt-Token-Kosten.
  - `X-AcademicAI-Output-Cost`: Berechnete Completion-Token-Kosten.
  - `X-AcademicAI-Prompt-Tokens`: Tatsächliche Prompt-Tokens.
  - `X-AcademicAI-Completion-Tokens`: Tatsächliche Completion-Tokens.
  - `X-AcademicAI-Cost-Currency`: Währung (`"EUR"`).
  - `X-AcademicAI-Cost-Estimated`: `"true"` bei gestaffelten oder fehlenden Modellpreisen, sonst `"false"`.

### Persistenter lokaler Aggregator (`LocalCostStore`, `CostAggregationBucket`)
- **Genau-einmal-Abrechnung (`record_request`):**
  - Bucht Tokens und Kosten für non-streaming und streaming Anfragen exakt einmal.
- **Aggregationsdimensionen:**
  - `all_time`: Gesamtsummen seit Aufzeichnung.
  - `today`: Aufgeschlüsselt nach aktuellem UTC-Tag (`YYYY-MM-DD`).
  - `this_month`: Aufgeschlüsselt nach aktuellem UTC-Monat (`YYYY-MM`).
  - `by_model`: Aufgeschlüsselt nach Modell-Identifikator.
  - `by_client`: Aufgeschlüsselt nach anonymisiertem SHA-256 Client-Hash (`client_<hash[:8]>`).
- **Datenschutzkonformer Ringpuffer (`recent_requests`):**
  - Fester Puffer der letzten `LOCAL_COST_HISTORY_LIMIT` Anfragen (Default: 500).
  - Enthält **ausschließlich** Abrechnungs-Metadaten (`timestamp`, `model`, `client_id`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `request_cost`, `currency`, `is_estimated`).
  - **Streng verboten und technisch ausgeschlossen:** Keine Speicherung von Prompts, Completions, Tools oder Roh-API-Keys.
- **Atomare Dateipersistenz:**
  - Atomares Schreiben nach `LOCAL_COST_CACHE_FILE` (Default: `data/local_cost_cache.json`) via temporäre Datei, `os.fsync` und `os.replace` mit Windows-Sharing-Retry-Logik.

---

## 8. Runtime-Lifecycle & Health-Zustand ([`academicai/runtime.py`](../../academicai/runtime.py))

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
- Führt bei aktiviertem Check (`HEALTH_CHECK_BACKEND=True`) einen synchronen HTTP-GET-Aufruf gegen den AcademicAI-Endpunkt `/api/v1/llm/models` via `httpx.Client` aus (Timeout konfiguriert über `HEALTH_CHECK_TIMEOUT_SECONDS`, Default: 2.0s).
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

## 9. Logging-Infrastruktur & Handler-State ([`academicai/logging_config.py`](../../academicai/logging_config.py))

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

---

## 10. Humanization & Zielkanal-Klassifizierung ([`academicai/humanization.py`](../../academicai/humanization.py))

Das Modul [`academicai/humanization.py`](../../academicai/humanization.py) kapselt Heuristiken zur Unterscheidung menschlicher Chat-Kanäle von maschinellen API-Aufrufen, die Extraktion des letzten User-Prompts, die Konstruktion von Prompts für den zweiten LLM-Pass und die asynchrone Ausführung des Humanisierungs-Passes:

### Zielkanal-Erkennung (`is_human_readable_target`, `_is_human_readable_target`)
- **Heuristische Kanal-Klassifizierung:**
  - Erkennt typische Messenger- und Chat-Kanäle im System-Kontext (`whatsapp`, `telegram`, `signal`, `imessage`, `discord`, `slack`, `googlechat`, `irc`, `webchat`, sowie Gruppen-/Direktchat-Typen).
  - Erkennt OpenClaw-spezifische User-Envelopes mit Metadaten (`conversation info (untrusted metadata)`, `"is_group_chat": true/false`, `"conversation_label":`, `"sender": "+..."`) oder Standard-Session-Prompts (`you are a personal assistant running inside openclaw.`).
- **Maschinen-Trigger-Ausschluss:**
  - Schließt maschinelle Jobs (z.B. automatisierte Cron-Trigger `[cron:...`) explizit aus (`False`), selbst wenn Messenger-Marker im System-Prompt vorhanden sind.

### User-Text-Extraktion (`last_user_text`, `_last_user_text`)
- Durchsucht die Nachrichten-Historie rückwärts nach der letzten Nachricht mit `role == "user"`.
- Verwendet `extract_text_content` aus [`academicai/transformation.py`](../../academicai/transformation.py) zur robusten Extraktion sowohl aus Plain-Strings als auch aus Multi-Part-/Dict-Strukturen.

### Prompt-Konstruktion (`build_humanization_messages`, `_build_humanization_messages`)
- Baut eine 2-Turn-Nachrichtenliste für den optionalen zweiten LLM-Pass:
  - `system`: Klare Instruktion, strukturierte Tool-Ergebnisse in natürliche, fließende Antworten ohne JSON, Codeblöcke, Feldnamen oder Debug-Metadaten umzuformulieren.
  - `user`: Enthält die ursprüngliche Benutzerfrage (`Original user question`) und das strukturierte Werkzeugergebnis (`Structured/tool-derived result`) mit Fallback auf `"-"` bei leeren Eingaben.

### Asynchrone Pass-Ausführung (`run_humanization_pass`, `_run_humanization_pass`)
- Führt den zweiten Pass asynchron via `run_in_threadpool` über die `completion`-Funktion aus.
- Unterstützt sowohl synchrone Callables als auch asynchrone Coroutine-Funktionen.
- Robustes Logging und Fallback: Fängt Laufzeitfehler ab, protokolliert Warnungen (`log.warning`) und liefert `None` zurück (Fallback auf First-Pass-Inhalt).
- Behandelt leere Antworten oder Whitespace-Only-Strings fehlertolerant durch Rückgabe von `None`.

### Dynamische Attributauflösung & Rückwärtskompatibilität
- **Dynamische Konfiguration:**
  - Prüft Attribute auf dem `server`-Modul (`HUMANIZATION_MODEL`, `HUMANIZATION_TEMPERATURE`, `academicai.completion`) vor dem Fallback auf `academicai.config` bzw. `academicai.completion`.
  - Stellt sicher, dass bestehende Test-Suiten mit `monkeypatch.setattr(server, ...)` ohne Anpassung funktionieren.
- **Server- und Package-Re-Exports:**
  - `server.py` und `academicai/__init__.py` exportieren alle 8 Funktionen und Aliase (`build_humanization_messages`, `_build_humanization_messages`, `is_human_readable_target`, `_is_human_readable_target`, `last_user_text`, `_last_user_text`, `run_humanization_pass`, `_run_humanization_pass`).

---

## 11. Application Factory & Modern Lifespan ([`academicai/app.py`](../../academicai/app.py))

Das Modul [`academicai/app.py`](../../academicai/app.py) kapselt die FastAPI-Anwendungsinstanziierung, den modernen ASGI-Lifespan-Handler sowie das zentrale Routing aller öffentlichen und internen HTTP-Endpunkte:

### Modern Lifespan Management (`lifespan`)
- **Ersatz für `@app.on_event`:** Ersetzt die veralteten `@app.on_event("startup")` und `@app.on_event("shutdown")`-Dekoratoren vollständig durch einen standardkonformen Starlette/FastAPI `@asynccontextmanager async def lifespan(application: FastAPI)`-Handler.
- **Startup Phase:** Führt `validate_config()` (Integritätsprüfung des API-Keys) und `write_pid_file()` (Registrierung der Prozess-ID) aus, bevor Anfragen verarbeitet werden.
- **Shutdown Phase:** Ruft `cleanup_pid_file()` im `finally`-Block auf, um die PID-Datei bei Beendigung des Prozesses sauber zu entfernen.

### Application Factory (`create_app`)
- Instanziiert und konfiguriert die `FastAPI`-Applikation mit Metadaten (`title="AcademicAI Proxy"`, `version="1.0.0"`) und bindet den `lifespan`-Kontextmanager ein.
- Registriert alle HTTP-Routen deklarativ:
  - `GET /health` $\to$ `health`
  - `GET /internal/cost-status` $\to$ `cost_status` (authentifiziert via `verify_key`)
  - `GET /v1/models` $\to$ `list_models` (authentifiziert via `verify_key`)
  - `POST /v1/chat/completions` $\to$ `chat_completions` (Request-Guards, Rate-Limiting, Tool-Emulation, Streaming, Humanisierung)
- Stellt eine modulweite Singleton-Instanz `app = create_app()` bereit.

### Dynamische Attributauflösung (`_get_setting`, `_INITIAL_DEFAULTS`)
- Kapselt dynamische Konfigurations- und Funktionsauflösung zur Unterstützung bestehender Test-Suiten:
  - Erkennt `monkeypatch.setattr(server, ...)` und `monkeypatch.setattr(academicai.app, ...)` präzise durch Abgleich mit `_INITIAL_DEFAULTS`.
  - Gewährleistet nahtlose Weiterleitung von `API_KEY`, Schutzgrenzen, Modelllisten und Completion-Funktionen.

### Entkoppelte Tool-Guards & Text-Extraktion
- **`academicai/tool_emulation.py`:** Beherbergt `apply_post_tool_guard` / `_apply_post_tool_guard` (Follow-up-Stabilisierung nach Tool-Ergebnissen).
- **`academicai/transformation.py`:** Konsolidiert `extract_text_content` / `_extract_text_content` (Normalisierung von Strings und Multipart-Dicts) als Single Source of Truth.

---

## 12. CLI-Entrypoint & Kompatibilitätsschicht ([`server.py`](../../server.py))

Das Root-Skript [`server.py`](../../server.py) wurde im Zuge des Refactorings zu einem reinen, schlanken Einstiegspunkt und Kompatibilitäts-Layer kontrahiert (< 200 Zeilen):

- **Reiner CLI-Runner:**
  - Lädt Umgebungsvariablen (`load_dotenv()`).
  - Zeigt das Start-Banner mit Port, Auth-Status, Log-Rotation, Humanisierungs-Status und Request-Limits an.
  - Startet den ASGI-Server via `uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")`.
- **Vollständige Rückwärtskompatibilität:**
  - Re-exportiert alle wesentlichen Symbole, Einstellungen und Hilfsfunktionen aus den Domänenmodulen (`academicai.app`, `academicai.config`, `academicai.request_guards`, `academicai.tool_emulation`, `academicai.transformation`, `academicai.cost_monitoring`, `academicai.runtime`, `academicai.logging_config`, `academicai.humanization`).
  - Stellt sicher, dass bestehende Test-Fixtures (`server.app`), Test-Monkeypatches (`server.MAX_MESSAGES`, `server.academicai.completion`, `server._check_backend_health` etc.) und externe Aufrufer ohne Codeänderung fehlerfrei funktionieren.
- **Azyklische Modulstruktur:**
  - `server.py` enthält keinerlei eigene Domänenlogik mehr; alle Abhängigkeiten fließen unidirektional von `server.py` in die `academicai`-Module.

---

## 13. OpenAI Responses API & Codex Wire Protocol ([`academicai/responses.py`](../../academicai/responses.py))

Das Modul [`academicai/responses.py`](../../academicai/responses.py) kapselt die Request-Normalisierung, SSE-Wire-Event-Generierung und Token-Usage-Berechnung für die OpenAI Responses API (insbesondere für OpenAI Codex CLI und Desktop):

### Inbound-Normalisierung (`normalize_responses_request`)
- **`instructions` $\to$ System-Prompt:**
  - Wandelt den Top-Level-String `instructions` in eine `{"role": "system", "content": instructions}` Nachricht am Kopf der Nachrichtenliste um.
- **`input`-Normalisierung:**
  - Unterstützt String-Eingaben (`input: "..."` $\to$ User-Message).
  - Array von Objekten:
    - `type: "message"`: Behält Rolle bei (`developer` wird in `transformation.py` wie `system` behandelt). Extrahiert Text aus `input_text` oder `output_text` Content-Parts.
    - `type: "function_call"`: Konvertiert vorherige Codex-Toolaufrufe in Assistant-Nachrichten mit `tool_calls` (`call_id`, `name`, `arguments`).
    - `type: "function_call_output"`: Konvertiert Werkzeugergebnisse aus der lokalen Codex-Sandbox in `role: "tool"` Nachrichten (`tool_call_id=call_id`, `content=output`).
- **Tool-Definitionen:**
  - Unterstützt sowohl flache Tool-Definitionen (wie von Codex gesendet: `{"type": "function", "name": "...", "description": "...", "parameters": {...}}`) als auch verschachtelte OpenAI-Tools (`{"type": "function", "function": {...}}`).

### Outbound-Serialisierung (`build_responses_output`)
- Erzeugt ein nicht-streamendes Responses-API-JSON-Objekt:
  - `id`: Eindeutige ID mit Präfix `resp_...`.
  - `object`: `"response"`.
  - `status`: `"completed"`.
  - `output`: Liste von Output-Items:
    - Textantwort: `[{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "..."}]}]`
    - Tool-Calls: `[{"type": "function_call", "call_id": "...", "name": "...", "arguments": "..."}]`
  - `usage`: Vollständiges Token-Accounting.

### SSE-Streaming-Events (`build_responses_sse_events`)
- Generiert standardkonforme OpenAI Responses Server-Sent Events für Codex:
  1. `response.created` (initialer Response-Envelope)
  2. `response.in_progress` (Bearbeitungsstatus)
  3. Pro Output-Item:
     - Text: `response.output_item.added` (`type: "message"`), `response.content_part.added`, gefolgt von `response.text.delta`-Events pro Token-Chunk, und `response.output_item.done`.
     - Tool-Call: `response.output_item.added` (`type: "function_call"`), `response.function_call_arguments.delta`-Events mit JSON-Argumenten, und `response.output_item.done`.
  4. `response.completed` (finaler Response-Envelope mit `usage` und `status: "completed"`).

### Token-Usage & Rust-Deserializer-Kompatibilität (`_normalize_responses_usage`)
- Die OpenAI Codex CLI (Rust-basiert) verlangt im `usage`-Objekt von `response.completed` strikt die Schlüssel `input_tokens` und `output_tokens`.
- `_normalize_responses_usage` mappt die Token-Zahlen konsistent:
  - `input_tokens` $\leftrightarrow$ `prompt_tokens`
  - `output_tokens` $\leftrightarrow$ `completion_tokens`
  - `total_tokens`
- Liefert alle 5 Schlüssel aus, wodurch sowohl die Codex CLI als auch Standard-OpenAI-Clients fehlerfrei deserialisieren können.

---

## 14. Modell-Konnektivitäts- & Diagnose-CLI ([`test_models_connectivity.py`](../../test_models_connectivity.py))

Das Root-Skript [`test_models_connectivity.py`](../../test_models_connectivity.py) dient als primäres Werkzeug zur Diagnose von Netzwerkverbindungen, Modellverfügbarkeit und Upstream-Fehlern:

- **Dual-Mode-Architektur:**
  - **Local-Proxy-Modus (Default):** Testet den lokal laufenden Proxy auf Port 11435 (`/v1/models` und `/v1/chat/completions`) über non-streaming und SSE-Streaming.
  - **Upstream-Direktmodus (`--upstream` / `-u`):** Testet unter Umgehung des lokalen Proxies direkt gegen das AcademicAI-Backend (`/api/v1/llm/models` und `/api/v1/llm/chat`). Ermöglicht sofortige Isolation zwischen lokalen Proxy-Problemen und Upstream-Fehlern (z.B. abgelaufene Credentials, Cost Limits).
- **Strukturierte Fehler-Extraktion (`_extract_error_message`):**
  - Erkennt sowohl OpenAI-kompatible Fehler (`{"error": {"message": ...}}`) als auch tief geschachtelte AcademicAI-Fehler (`{"meta": {"error": {"message": ...}}}`).
  - Verhindert das irreführende Maskieren von Kontingentfehlern als `"Backend 500"`; zeigt stattdessen z.B. `[FAIL] HTTP 429: AcademicAI Cost Limit Reached: API Client Error: Cost limit reached`.
- **Modell-Listing (`--list` / `-l`):**
  - Zeigt alle verfügbaren Modelle tabellarisch mit Kontextgröße, maximalem Token-Limit und normalisierten Preisen in `€/1M` an, ohne Test-Prompts abzufeuern.
- **Selektives Filtern (`--model <name>` / `-m <name>`):**
  - Ermöglicht gezieltes Testen einzelner Modelle oder Modell-Familien (z.B. `-m gpt-5-mini` oder `-m claude`).

