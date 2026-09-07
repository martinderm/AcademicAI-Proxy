# AcademicAI Proxy

OpenAI-compatible proxy for AcademicAI.
It exposes AcademicAI models on a local OpenAI-style API (default: `http://127.0.0.1:11435/v1`).

## Status

- Chat completions: ✅
- Model list endpoint: ✅
- Health endpoint: ✅
- Cost status endpoint: ✅
- Tool-call emulation (JSON-mode with TypeScript signatures & JSON repair): ✅
- SSE-style streaming emulation: ✅
- Daily Log Rotation (30 days retention): ✅
- E2E Test Port Isolation (runs on port 11436): ✅
- Automatic Prompt Caching Compatibility (Azure prefix caching): ✅
- Modular Domain Architecture & Modern ASGI Lifespan (`academicai.app`): ✅

### Caching and Costs Status

- **Automatic Prefix Caching**: ✅ Supported natively. The proxy is aligned to merge system instructions at the very beginning of the first user message, maximizing Azure OpenAI prefix cache hit rates.
- **Cost/usage monitoring**: 🔴 (Disabled / Forbidden on the BOKU backend credentials - endpoint returns `403 Forbidden` due to tenant permissions).

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
   The proxy merges system instructions and tool definitions at the very beginning of the first user message. Because the BOKU backend runs on Azure OpenAI, stable prefix tokens (system context + tool signatures) trigger automatic KV-cache hits. This reduces Time-To-First-Token (TTFT) from seconds to milliseconds on repeated turns.
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

- `GET /health`
- `GET /internal/cost-status` (cost snapshot)
- `GET /v1/models`
- `POST /v1/chat/completions`

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

## Tests

Run smoke + functional tests:

```powershell
py -m pytest -q
py tests/test_tool_emulation.py
py -m pytest -q tests/test_post_tool_guard.py
py -m pytest -q tests/test_hardening_security_runtime.py
```

Optional:

```powershell
py tests/test_openclaw_style.py
```

## Project layout

```text
academicai-proxy/
  academicai/
    __init__.py
    app.py               # FastAPI application factory & ASGI lifespan
    auth.py              # BOKU authentication & header injection
    config.py            # Typed settings & environment parsing
    cost_monitoring.py   # Atomic cache & cost status
    errors.py            # Standardized OpenAI error mapping
    humanization.py      # Target channel detection & 2nd-pass rewriting
    logging_config.py    # Rotating file handlers & uvicorn wiring
    provider.py          # HTTP transport to BOKU backend
    request_guards.py    # Inbound payload validation & rate limiting
    runtime.py           # Process lifecycle & backend health checks
    tool_emulation.py    # TypeScript signatures, repair & post-guard
    transformation.py    # Message role normalization & text extraction SSOT
  docs/
    architecture/        # Concept & modularization architecture docs
    archive/             # Legacy archive artifacts
    system-map/          # ICM-aligned agent architecture map
    tenant-template/     # Environment templates
  server.py              # Slim CLI runner & compatibility layer
  start_server.ps1       # Controlled service startup
  stop_server.ps1        # Controlled service shutdown
  run_local_tests.ps1    # Offline and E2E test runner
  tests/                 # Comprehensive test suite (175+ tests)
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
