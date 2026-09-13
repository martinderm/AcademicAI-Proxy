# AcademicAI Proxy

OpenAI-compatible proxy for [AcademicAI](https://www.acomarket.at/de/portfolio/projekte/academic-ai) (die KI-Initiative für Österreichs Universitäten im Rahmen des ACOmarket-Portfolios).
It exposes AcademicAI models on a local OpenAI-style API (default: `http://127.0.0.1:11435/v1`).

## Status

- Chat completions: ✅
- OpenAI Responses API (`POST /v1/responses` for Codex CLI & Desktop): ✅
- Model list endpoint: ✅
- Health endpoint: ✅
- Cost status endpoint: ✅
- Local Request Cost Calculation: ✅ (Autonomous Decimal cost accounting, pricing cache & aggregation)
- Tool-call emulation (JSON-mode with TypeScript signatures & JSON repair): ✅
- SSE-style streaming emulation: ✅
- Daily Log Rotation (30 days retention): ✅
- E2E Test Port Isolation (runs on port 11436): ✅
- Automatic Prompt Caching Compatibility (Azure prefix caching): ✅
- Modular Domain Architecture & Modern ASGI Lifespan (`academicai.app`): ✅
- OpenCode & OpenChamber Web/Mobile Harness (Tailscale): ✅ (see [docs/opencode-openchamber.md](docs/opencode-openchamber.md))

### Caching and Costs Status

- **Automatic Prefix Caching**: ✅ Supported natively. The proxy is aligned to merge system instructions at the very beginning of the first user message, maximizing Azure OpenAI prefix cache hit rates.
- **Autonomous Local Cost Calculation**: ✅ Fully operational. The proxy dynamically caches model pricing from `/api/v1/llm/models`, calculates exact request costs via `Decimal` arithmetic, injects standardized response headers, and maintains persistent local aggregations (`today`, `this_month`, `all_time`, `by_model`, `by_client`).
- **AcademicAI Backend Cost Endpoint (`/api/v1/cost/`)**: 🔴 (Returns `403 Forbidden` due to tenant permissions `ACCESS_API_MONITOR_CREDIT`). Local tracking completely bypasses this limitation.

## Why this proxy exists

AcademicAI does not provide native OpenAI function-calling/tool-calling in the same way OpenAI-compatible clients expect.
This proxy emulates the tool flow so orchestrators (e.g. OpenClaw) can still run tools reliably.

### Limits of tool-call emulation (plain language)

Tool-calling is **simulated**, not native. That means the model is guided via prompt + JSON parsing,
not by a backend-level function-calling engine.

In practice this works well, but there are limits:

- behavior is probabilistic (occasionally the model may answer in JSON style instead of ideal natural text)
- extra guardrails are needed to avoid unnecessary repeated tool calls
- reliability is generally lower than true native tool-calling APIs

So: good for practical use, but not mathematically deterministic.

### Efficiency and latency mechanics (why it is fast)

Users often observe that tool calling through this proxy feels surprisingly fast. This is driven by three specific architectural choices:

1. **Azure OpenAI KV-Prefix Caching:**  
   The proxy merges system instructions and tool definitions at the very beginning of the first user message. Because the AcademicAI backend runs on Azure OpenAI, stable prefix tokens (system context + tool signatures) trigger automatic KV-cache hits. This reduces Time-To-First-Token (TTFT) from seconds to milliseconds on repeated turns.
2. **Single-Pass Minimal Output (JSON Mode):**  
   Instead of a two-pass router or conversational tool descriptions, the backend is invoked in `response_format: {type: "json_object"}`. Modern models produce minimal JSON without pleasantries (`{"action": "tool_call", ...}`), emitting only 25–40 tokens per call.
3. **Zero Heavy Framework Overhead:**  
   Direct asynchronous HTTP transport via `httpx` and Python standard library parsing eliminates the latency overhead of heavy abstraction layers.

### Potential remaining problems and edge cases

While reliable for everyday agent tasks, emulating function calling over a text-only backend comes with structural trade-offs:

1. **Schema Compression Trade-offs:**  
   To prevent context window explosion when dozens of tools are registered, tool schemas are compressed into high-density TypeScript-style signatures (`_compact_tool_def`), featuring concise enum unions, typed arrays (`string[]`, `number[]`), defaults, and shallow objects (`{query, tags}`). Highly complex, deeply nested JSON schemas requiring deep object trees may still experience reduced parameter precision compared to native OpenAI function calling engines.
2. **`tool_choice` Prompt Enforcement:**  
   While the proxy enforces `tool_choice: "required"` and specific tool targets by strictly forbidding `{"action": "respond"}` in system prompts and reminders, enforcement happens at the prompt layer rather than the engine sampler level.
3. **Multi-Turn Role Flattening (`role: "tool"`):**  
   The underlying backend only accepts `user` and `assistant` roles. Tool results are flattened into user turns with standardized `<tool_result id="..." name="...">` XML tags. In deep multi-step loops (5+ sequential tool executions), this conversational history can sometimes cause attention drift, which the post-tool guard mitigates.
4. **Probabilistic vs. Deterministic Parsing & JSON Repair:**  
   Unlike native APIs where tool arguments are constrained by grammar-based token samplers, the model generates raw JSON text. The proxy pairs a multi-tier fallback parser (direct parse → markdown codeblock extraction → bracket-depth counter) with automatic JSON-repair sanitization (trailing comma stripping, unescaped control character leniency, and path escape fixes).

## Endpoints

- `GET /health` — Service & backend health check
- `GET /internal/cost-status` — Cached cost snapshot
- `GET /v1/models` — Dynamic model discovery
- `POST /v1/chat/completions` — Standard OpenAI Chat Completions API
- `POST /v1/responses` — OpenAI Responses API for OpenAI Codex CLI & Desktop

## Available Models

The proxy dynamically discovers and validates available models from the AcademicAI backend via `GET /v1/models`. Typical models supported include:

| Family | Model IDs | Key Features |
| :--- | :--- | :--- |
| **OpenAI** | `gpt-4o`, `gpt-4o-mini`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.2`, `gpt-5.5`, `o3` | Emulated tool-calling, Azure KV prefix caching, reasoning parameters |
| **Anthropic** | `claude-opus-4-6`, `claude-opus-4-8` | 1M token context window, deep reasoning |
| **Google** | `gemini-3.5-flash`, `gemini-3.1-flash-lite`, `gemini-3.1-pro-preview`, `gemini-2.5-pro` | 1M token context window, multimodal capabilities |
| **Perplexity** | `sonar-pro`, `sonar-reasoning-pro` | Built-in search and citations |
| **Mistral** | `Mistral-Large-3` | 256k context window |


## Lokale Kostenberechnung (Local Cost Calculation)

Da der AcademicAI-Endpunkt `/api/v1/cost/` für Standard-API-Clients `403 Forbidden` (`ACCESS_API_MONITOR_CREDIT`) zurückgibt, führt der Proxy eine autonome lokale Kostenberechnung pro Request durch.

### Preismechanik & Einheiten

AcademicAI liefert über den Endpunkt `/api/v1/llm/models` die Kosteninformationen pro Modell in der Datenstruktur `costs`.
- **Maßeinheit & Währung:** Preise in `costs` (für `input_tokens` und `output_tokens`) sind in **EUR pro 1.000 Tokens (1k Tokens)** angegeben (z. B. `0.00275` für `gpt-4o` Input = 0,00275 € / 1k Tokens).
- **Berechnungsformel:**
  $$\text{input\_rate} = \frac{\text{cost}}{1000}, \quad \text{output\_rate} = \frac{\text{cost}}{1000}$$
  $$\text{request\_cost} = (\text{prompt\_tokens} \times \text{input\_rate}) + (\text{completion\_tokens} \times \text{output\_rate}) + \text{per\_request\_cost}$$
- **Hochpräzise Arithmetik:** Sämtliche Berechnungen erfolgen mit Pythons `Decimal`-Modul, um Rundungsfehler bei Mikro-Cents vollständig zu vermeiden.
- **Kontext-Staffelung (Tiered Models):** Modelle mit mehreren Preisstufen (`gpt-5.5`, `gemini-2.5-pro`) nutzen die Basisstufe als Standard und markieren die Berechnung mit `X-AcademicAI-Cost-Estimated: true`.

### Response-Headers

Jede erfolgreiche Anfrage über `/v1/chat/completions` (sowohl non-streaming als auch streaming) sowie über `/v1/responses` liefert standardisierte Kosten- und Token-Header zurück:

| Header | Beschreibung | Beispiel |
| :--- | :--- | :--- |
| `X-AcademicAI-Request-Cost` | Gesamtkosten des Requests | `0.000825` |
| `X-AcademicAI-Input-Cost` | Berechnete Input-Token-Kosten | `0.00033` |
| `X-AcademicAI-Output-Cost` | Berechnete Output-Token-Kosten | `0.000495` |
| `X-AcademicAI-Prompt-Tokens` | Tatsächliche Prompt-Tokens | `120` |
| `X-AcademicAI-Completion-Tokens` | Tatsächliche Completion-Tokens | `45` |
| `X-AcademicAI-Cost-Currency` | Währung (Standard: EUR) | `EUR` |
| `X-AcademicAI-Cost-Estimated` | `true`, falls Modellpreise geschätzt/gestaffelt | `false` |

*Hinweis:* Standard-OpenAI-Response-Payloads bleiben 100 % unverändert und frei von proprietären Feldern, um die Kompatibilität mit Clients wie Cursor, Codex oder OpenClaw zu garantieren.

### Lokaler 24h Modellkatalog & Preistabelle (ModelCatalog)

Der Proxy erfasst alle verfügbaren Modelle und deren Preise einmalig vom Endpunkt `/api/v1/llm/models` und persistiert den Modellkatalog atomar als JSON-Datei in `data/model_catalog.json` mit einer **Lebensdauer (TTL) von 24 Stunden (86.400 Sekunden)**:
- **Zero-Latency Model Discovery (`GET /v1/models`):** Der Proxy liefert die Modell-Liste direkt aus dem im Speicher gehaltenen Katalog im Standard-OpenAI-Format (`to_openai_models_response()`). Dies eliminiert den 200–500ms langen Upstream-Roundtrip bei jedem Start von Agenten-Tools (OpenCode, Codex, OpenClaw) und bietet vollständige Offline-Resilienz gegen Upstream-Ausfälle.
- **Metadaten & Tokengrenzen:** Speichert neben Preisen auch `context_window` (z. B. bis zu 1.050.000 Tokens) und `output_token_limit` (`max_tokens`).
- **Startup ohne Latenz:** Beim Start des Proxies wird der Katalog sofort aus der lokalen JSON-Datei geladen.
- **Background Refresh:** Nach Ablauf der 24 Stunden wird der Refresh asynchron im Hintergrund ausgelöst, ohne den Client-Request zu blockieren.
- **Fehlertoleranz:** Sollte die AcademicAI-Modell-API temporär nicht erreichbar sein, greift der Proxy transparent auf den zuletzt gespeicherten Stand zurück.

### Lokale Aggregation & Status-Endpunkt (`/internal/cost-status`)

Der Proxy aggregiert die Kosten serverseitig in-memory und persistiert sie atomar in `data/local_cost_cache.json`.
Der Endpunkt `GET /internal/cost-status` liefert:
- `backend_cost_monitoring`: Status der AcademicAI-Kostenüberwachung (falls vorhanden).
- `local_cost_tracking`:
  - `all_time`: Gesamtkosten, Token-Summen und Request-Count.
  - `today`: Aggregation für den aktuellen Tag (UTC).
  - `this_month`: Aggregation für den aktuellen Monat (UTC).
  - `by_model`: Aufschlüsselung pro Modell-ID.
  - `by_client`: Aufschlüsselung nach anonymisiertem Client-Hash (`client_<sha256[:8]>`).
  - `model_catalog`: Status des Modellkatalogs (`models_cached`, `last_refreshed_at`, `is_stale`, `ttl_seconds`, `catalog_file`).
  - `recent_requests`: Ringpuffer der letzten 500 Requests (streng datenschutzkonform: nur Metadaten, keine Prompts, Completions oder API-Keys!).

### Konfigurationsoptionen

In der `.env` konfigurierbar:

| Variable | Typ | Default | Beschreibung |
| :--- | :--- | :--- | :--- |
| `ACADEMICAI_ENABLE_LOCAL_COST_TRACKING` | bool | `true` | Aktiviert die lokale Kostenberechnung & Header |
| `ACADEMICAI_COST_CURRENCY` | str | `EUR` | Währung für Abrechnung & Response-Header (Standard: EUR) |
| `ACADEMICAI_MODEL_CATALOG_FILE` | str | `data/model_catalog.json` | Pfad zum persistenten Modellkatalog |
| `ACADEMICAI_MODEL_CATALOG_TTL_SECONDS` | int | `86400` | Gültigkeitsdauer des Modellkatalogs in Sekunden (24 Stunden) |
| `ACADEMICAI_LOCAL_COST_CACHE_FILE` | str | `data/local_cost_cache.json` | Pfad zur lokalen Aggregationsdatei |
| `ACADEMICAI_LOCAL_COST_HISTORY_LIMIT` | int | `500` | Maximale Einträge im Ringpuffer der Request-Historie |



## Authentication

Client -> Proxy (Bearer):

- `Authorization: Bearer <YOUR_PROXY_API_KEY>`

Proxy -> AcademicAI backend:

- `X-Client-ID: <ACADEMICAI_CLIENT_ID>`
- `X-Client-Secret: <ACADEMICAI_CLIENT_SECRET>`

Configure these values in `.env` (never commit real secrets).
Startup fails fast when `ACADEMICAI_PROXY_API_KEY` is missing, insecure, or too short.

## Tenant separation (generic repo vs external tenant config)

Keep repository content generic. Put tenant-specific values outside the repository:

- `.env` with endpoint, client ID/secret, proxy API key

Templates are provided under `docs/tenant-template/`.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env
```

Run:

```powershell
py server.py
# or
.\start_server.ps1
# optional controlled stop
.\stop_server.ps1
```

## OpenAI Codex Integration

Der Proxy unterstützt [OpenAI Codex](https://github.com/openai/codex) (CLI und Desktop) über den standardkonformen Endpunkt `POST /v1/responses` (`wire_api = "responses"`).

Konfiguration in `~/.codex/config.toml` bzw. `%USERPROFILE%\.codex\config.toml`:

```toml
model = "gpt-4o"
model_provider = "academicai"

[model_providers.academicai]
name = "AcademicAI"
base_url = "http://127.0.0.1:11435/v1"
wire_api = "responses"
env_key = "ACADEMICAI_PROXY_API_KEY"
supports_websockets = false
```

Ausführliche Details zu Tools, Multi-Turn-Roundtrips und Modellwahl findest du in [`docs/codex.md`](docs/codex.md).


## OpenCode & OpenChamber Integration

Der Proxy dient als primäres LLM-Backend für [OpenCode](https://opencode.ai) und das Web-/PWA-Frontend [OpenChamber](https://github.com/openchamber/web) via `@ai-sdk/openai-compatible`.

- **Konfigurationsanleitung:** Vollständige Einrichtung und Tailscale-Sicherheitsarchitektur siehe [`docs/opencode-openchamber.md`](docs/opencode-openchamber.md).
- **Modell-Charakteristiken:** In `opencode.json` sollten `limit.context`, `limit.output`, `tool_call` und `reasoning` stets explizit hinterlegt werden, um konservative Fallbacks (4k/8k) zu vermeiden.
- **Empfehlung: 256k-Kontextgrenze:** Für Modelle mit 1M+ Backend-Kapazität (`gpt-5.5`, `claude-opus-4-8`, `gemini-3.5-flash`) empfiehlt sich ein Cap auf `context: 256000` (256k Tokens). Das schützt das Token-Budget vor versehentlicher Erschöpfung bei langen Sessions, hält Antwortzeiten kurz und bietet mit ~800–1.000 Buchseiten Text mehr als genug Raum.


## Local test environment

The repository now includes a separate local test setup:

- `.env.localtest` for local test defaults
- `.env.localtest.example` as template
- `.\start_test_server.ps1` to start the proxy with local test settings
- `.\run_local_tests.ps1 -Mode offline` for local/offline regression tests
- `.\run_local_tests.ps1 -Mode e2e` for end-to-end proxy tests against AcademicAI

Offline mode:

- does not require real AcademicAI backend credentials
- validates hardening, request validation, guard logic, and humanization helpers

E2E mode:

- requires `ACADEMICAI_BASE_URL`, `ACADEMICAI_CLIENT_ID`, and `ACADEMICAI_CLIENT_SECRET` in `.env.localtest`
- uses `ACADEMICAI_TEST_PROXY_API_KEY` / `ACADEMICAI_TEST_BASE_URL` from `.env.localtest` for the local proxy side

Quick start for local testing:

```powershell
.\run_local_tests.ps1 -Mode offline
```

## Model connectivity & diagnostic CLI

The repository includes a versatile CLI tool [`test_models_connectivity.py`](test_models_connectivity.py) to inspect registered models, pricing, and connectivity:

```bash
# 1. Quick overview of available upstream models & pricing (no completion tokens used):
python test_models_connectivity.py --upstream --list

# 2. Test a single model directly against the AcademicAI upstream backend:
python test_models_connectivity.py --upstream -m gpt-5-mini

# 3. Test all models via the running local proxy (port 11435):
python test_models_connectivity.py

# 4. Test a specific model family via local proxy:
python test_models_connectivity.py -m claude
```

Key features:
- **Dual-Mode**: Test through local proxy or directly against upstream BOKU AcademicAI API (`--upstream` / `-u`).
- **Precise Error Diagnostics**: Formats OpenAI-style `error.message`, BOKU `meta.error.message` (e.g. `Cost limit reached`), and FastAPI details without masking.
- **Model Listing**: Formats context window, output token limit, and normalized input/output costs in €/1M tokens (`--list` / `-l`).


## Runtime hardening defaults

- `ACADEMICAI_PROXY_API_KEY` is mandatory and must be changed from insecure placeholders.
- API key values shorter than 16 chars are rejected at startup.
- Debug dumps are secret-redacted (`Authorization`, tokens, client secrets).
- `POST /v1/chat/completions` enforces request shape and size limits.
- Per-minute rate limiting is enabled by default (`ACADEMICAI_RATE_LIMIT_PER_MINUTE=120`).
- `GET /health` includes backend status and reports `degraded` if backend check fails.

## Recommended workflow: let an AI agent install it from this GitHub URL

This repository is structured so both humans and LLM agents can install it reliably.
Recommended approach: give your coding agent the GitHub URL and ask it to perform setup + verification.

Suggested instruction you can paste to an agent:

```text
Install and verify this repository as a local service:
1) clone repo
2) create .env from .env.example
3) fill ACADEMICAI_BASE_URL, ACADEMICAI_CLIENT_ID, ACADEMICAI_CLIENT_SECRET, ACADEMICAI_PROXY_API_KEY
4) install dependencies
5) start server
6) verify /health and /v1/models with Bearer auth
7) run pytest
8) report final status and exact local run command
```

Minimum verification checks:

- `GET /health` returns `{"status":"ok"...}`
- `GET /v1/models` works with `Authorization: Bearer <ACADEMICAI_PROXY_API_KEY>`
- `py -m pytest -q` passes

## Supported request parameters (`POST /v1/chat/completions`)

Required:

- `model`
- `messages`

Forwarded when set:

- `stream`
- `temperature`
- `max_tokens`
- `max_completion_tokens`
- `frequency_penalty`
- `presence_penalty`
- `reasoning_effort`
- `verbosity`
- `seed`
- `stop`
- `response_format` (dict only; tool-mode forces `{ "type": "json_object" }`)
- `extra_body.tailoredAiId`

Tool emulation input (not forwarded natively):

- `tools` (alias: `functions`)
- `tool_choice`

Validation and protection behavior:

- Invalid JSON body: `400`
- Invalid request shape (e.g. wrong `messages` type): `422`
- Oversized payload/tool schema/message content: `413`
- Rate limit exceeded: `429`

## Supported request parameters (`POST /v1/responses`)

OpenAI Responses API endpoint designed for OpenAI Codex CLI and Desktop (`wire_api = "responses"`):

Required:

- `model` (e.g. `gpt-4o`, `gpt-4o-mini`, etc.)
- `input` (list of structured input items or plain text string)

Supported input item types:

- `type: "message"`: Standard conversational turn. Roles supported: `user`, `assistant`, `system`, `developer`. Text content can be a plain string or array of parts (`type: "input_text"` or `type: "output_text"`).
- `type: "function_call"`: Tool calls emitted in prior turns (`call_id`, `name`, `arguments`).
- `type: "function_call_output"`: Output returned from client-side execution in Codex's local sandbox (`call_id`, `output`).

Optional:

- `instructions`: Top-level system prompt instructions.
- `tools`: List of tool definitions. Supports both flat schema (`{"type": "function", "name": "...", "parameters": {...}}`) and nested OpenAI schema (`{"type": "function", "function": {...}}`).
- `stream`: Boolean (`true` for SSE streaming wire events, `false` for non-streaming JSON).
- `temperature`: Temperature override.
- `max_output_tokens`: Maximum completion tokens to generate.

Token Accounting & Metadata:

- Emits `input_tokens` and `output_tokens` (strictly required by OpenAI Codex CLI's Rust parser) as well as `prompt_tokens`, `completion_tokens`, and `total_tokens` in `usage`.


## Default behavior tuning (env)

These defaults apply only when the client did not set the field explicitly.

- `ACADEMICAI_DEFAULT_CHAT_TEMPERATURE=0.6`
- `ACADEMICAI_DEFAULT_CHAT_VERBOSITY=medium` (gpt-5* only)
- `ACADEMICAI_DEFAULT_TOOL_TEMPERATURE=0.1`
- `ACADEMICAI_DEFAULT_TOOL_VERBOSITY=low` (gpt-5* only)
- `ACADEMICAI_DEFAULT_TOOL_REASONING_EFFORT=low` (gpt-5* only)

Rule:

- Human conversation without tool mode -> chat defaults
- Tool mode (`tools`/`functions` present) -> tool defaults
- Explicit client fields always win

### Optional humanization pass (recommended for chat UX)

You can enable a second LLM pass that rewrites structured/tool-derived output
into natural human text.

- Active only for human channels
- Active only in tool mode
- Skipped when the model emits an actual tool call (`finish_reason: tool_calls`)

Env flags:

- `ACADEMICAI_ENABLE_HUMANIZATION_PASS=true|false`
- `ACADEMICAI_HUMANIZATION_MODEL=<optional override>` (default: same model)
- `ACADEMICAI_HUMANIZATION_TEMPERATURE=0.2`

### Optional cost monitoring (currently untested in this repo setup)

Cost monitoring is **disabled by default** and does nothing unless explicitly enabled.
No automatic `/api/v1/cost` calls are made when disabled.

Env flags:

- `ACADEMICAI_ENABLE_COST_MONITORING=true|false` (default: `false`)
- `ACADEMICAI_COST_CACHE_FILE=./cost_cache.json`
- `ACADEMICAI_COST_CACHE_TTL_SECONDS=600`
- `ACADEMICAI_COST_REFRESH_TIMEOUT_SECONDS=8`

Behavior when enabled:

- Proxy lazily/background-refreshes cost cache from AcademicAI `GET /api/v1/cost`.
- Adds response headers on chat completions (`X-AcademicAI-Total-Cost`, `X-AcademicAI-Total-Clients`, `X-AcademicAI-Cost-Entries`, `X-AcademicAI-Cost-Updated-At`, `X-AcademicAI-Cost-Stale`).
- Exposes `GET /internal/cost-status`.

Important prerequisites (per AcademicAI API docs):

- API client permission `ACCESS_API_MONITOR_CREDIT` is required for `/api/v1/cost`.
- Without that permission, the endpoint returns `403` and cache stays empty/stale.

## Notes for OpenClaw users

If you want proxy defaults to control style, do **not** hard-set these in OpenClaw for this provider:

- `temperature`
- `verbosity`
- `reasoning_effort`

## Tool-call emulation summary

1. Proxy reads `tools` from request
2. Injects tool schema/instructions into prompt
3. Forces JSON response mode
4. Parses model JSON into either:
   - single tool call: `{"action":"tool_call",...}`
   - multi-step tool calls: `{"action":"tool_calls","calls":[...]}`
   - normal assistant text (`{"action":"respond",...}`)
5. Converts tool call(s) into OpenAI `tool_calls` response (`finish_reason: tool_calls`)
6. Upstream orchestrator executes tool and sends `role=tool` follow-up

### Post-tool guard (stability improvement)

When tool mode is active and the latest message already has `role=tool`,
the proxy injects a short guard instruction that prefers a final user-facing
answer and discourages unnecessary additional tool calls.

This reduces accidental re-tooling loops while still allowing another tool call
if the latest tool result is clearly incomplete.

## API Quick Reference (cURL)

### 1. Health Check
```bash
curl -s http://127.0.0.1:11435/health
```

### 2. List Available Models
```bash
curl -s -H "Authorization: Bearer <ACADEMICAI_PROXY_API_KEY>" \
  http://127.0.0.1:11435/v1/models
```

### 3. Chat Completions (`POST /v1/chat/completions`)
```bash
curl -s -H "Authorization: Bearer <ACADEMICAI_PROXY_API_KEY>" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:11435/v1/chat/completions \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### 4. OpenAI Responses API (`POST /v1/responses`)
```bash
curl -s -H "Authorization: Bearer <ACADEMICAI_PROXY_API_KEY>" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:11435/v1/responses \
  -d '{
    "model": "gpt-4o",
    "instructions": "You are a concise assistant.",
    "input": [
      {"type": "message", "role": "user", "content": "Say OK"}
    ]
  }'
```

## Tests

Run the complete offline regression test suite (186 unit and integration tests):

```powershell
.\run_local_tests.ps1 -Mode offline
```

Or via pytest directly:

```powershell
pytest -q
```

To run end-to-end tests against the live AcademicAI backend (requires credentials configured in `.env.localtest`):

```powershell
.\run_local_tests.ps1 -Mode e2e
```


## Project layout

```text
academicai-proxy/
  academicai/
    __init__.py
    app.py               # FastAPI application factory & ASGI lifespan
    auth.py              # AcademicAI authentication & header injection
    config.py            # Typed settings & environment parsing
    cost_calculation.py  # ModelCatalog & Decimal request cost calculation
    cost_monitoring.py   # Atomic cache & cost status
    errors.py            # Standardized OpenAI error mapping
    humanization.py      # Target channel detection & 2nd-pass rewriting
    local_cost_tracker.py # LocalCostStore, aggregations & ring-buffer history
    logging_config.py    # Rotating file handlers & uvicorn wiring
    provider.py          # HTTP transport to AcademicAI backend
    request_guards.py    # Inbound payload validation & rate limiting
    responses.py         # OpenAI Responses API normalization & SSE serialization
    runtime.py           # Process lifecycle & backend health checks
    tool_emulation.py    # TypeScript signatures, repair & post-guard
    transformation.py    # Message role normalization & text extraction SSOT
  docs/
    architecture/        # Concept & modularization architecture docs
    archive/             # Legacy archive artifacts
    codex.md             # OpenAI Codex CLI & Desktop setup guide
    opencode-openchamber.md # OpenCode & OpenChamber Web/Mobile (Tailscale) setup guide
    system-map/          # ICM-aligned agent architecture map
    tenant-template/     # Environment templates
  server.py              # Slim CLI runner & compatibility layer
  start_server.ps1       # Controlled service startup
  stop_server.ps1        # Controlled service shutdown
  run_local_tests.ps1    # Offline and E2E test runner
  tests/                 # Comprehensive test suite (225+ tests)
  requirements.txt
  README.md
```

## Contributing / Upstream bugfixes

If your agent or team uses this proxy in production and you patch a bug,
please contribute it upstream so others benefit too:

- GitHub: <https://github.com/martinderm/AcademicAI-Proxy>

(If the repository URL changes, update this section accordingly.)

## License

MIT (see `LICENSE`).
