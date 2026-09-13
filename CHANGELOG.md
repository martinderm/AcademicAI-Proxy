# Changelog

## 0.8.1 - 2026-09-13

### 📋 Persistent 24h Model Catalog & Zero-Latency Discovery
- **Unified Local Model Catalog (`academicai/cost_calculation.py`)**:
  - Expanded the model pricing cache into a persistent 24h **Local Model Catalog** (`data/model_catalog.json`, `ModelCatalog`), capturing full model metadata (`context_window` / `contextWindow`, `output_token_limit` / `outputTokenLimit`) alongside pricing structures (`costs`).
  - Single Source of Truth (SSOT) for registered models, token limits, and pricing with automatic migration fallback from legacy `data/model_pricing_cache.json`.
- **Zero-Latency Model Discovery (`GET /v1/models`)**:
  - `AcademicAIProvider.get_models()` and `GET /v1/models` now serve directly from the in-memory `ModelCatalog` via `to_openai_models_response()`.
  - Eliminates the 200–500ms upstream network roundtrip on every client tool startup (OpenCode, Codex, OpenClaw), providing sub-millisecond responses and complete resilience against upstream network outages.
- **Unified Configuration & Backward Compatibility**:
  - Configurable via `ACADEMICAI_MODEL_CATALOG_FILE` and `ACADEMICAI_MODEL_CATALOG_TTL_SECONDS` (default: 86400s / 24h), while preserving `ACADEMICAI_MODEL_PRICING_CACHE_*` aliases.
  - Re-exports across `server.py` and `academicai/__init__.py` (`ModelCatalog`, `ModelEntry`, `get_model_catalog`, `ModelPricingCache`, `get_pricing_cache`).
  - Extended `/internal/cost-status` to report `model_catalog` status alongside `pricing_cache`.
- **Comprehensive Unit Tests**:
  - Extended `tests/test_cost_calculation_unit.py` (23 passing tests) and `tests/test_config.py` verifying catalog serialization, OpenAI response formatting, metadata extraction, and fallback paths.

## 0.8.0 - 2026-09-13

### 💰 Robust Autonomous Local Request Cost Calculation
- **Autonomous Request-Level Cost Accounting**: Combines model pricing metadata from `/api/v1/llm/models` with actual response token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`), completely eliminating dependence on the restricted `/api/v1/cost/` backend endpoint (`ACCESS_API_MONITOR_CREDIT` 403 Forbidden).
- **Dynamic Model Pricing Cache (`academicai/cost_calculation.py`)**:
  - Thread- and async-safe caching of `/api/v1/llm/models` pricing with configurable 24-hour TTL (default 86400s, `ACADEMICAI_MODEL_PRICING_CACHE_TTL_SECONDS`).
  - Atomic JSON disk persistence (`data/model_pricing_cache.json`) for zero-latency startup and offline/upstream outage fallback.
  - Empirical 1k-token pricing normalization: converts AcademicAI's per-1,000-token costs into per-token rates (`Decimal(cost) / 1000`).
  - High-precision `Decimal` arithmetic throughout to prevent floating-point drift on micro-cents.
  - Standardized default currency: `EUR` (configurable via `ACADEMICAI_COST_CURRENCY`).
  - **Currency SSOT Harmonization**: Top-level `currency: "EUR"` stored once as Single Source of Truth in `data/model_pricing_cache.json` with dynamic delegation from `ModelPricing`, eliminating redundant per-model repetitions.
  - Graceful degradation: cached fallback on network issues; fallback to base tier with `is_estimated = True` for tiered context models.
- **Diagnostic CLI Upgrade (`test_models_connectivity.py`)**:
  - Dual-mode architecture: test local running proxy (default) or bypass proxy to test directly against AcademicAI upstream (`--upstream` / `-u`).
  - Precise error formatting: extracts OpenAI-style `error.message` and upstream BOKU `meta.error.message` (e.g. `Cost limit reached`), preventing quota errors from being masked as generic `Backend 500`.
  - Model listing (`--list` / `-l`): tabular inspection of context window, output token limits, and normalized prices in `€/1M` without firing test completions.
  - Selective filtering (`--model <name>` / `-m <name>`): targeted testing of specific models or families.
- **Standardized Response Headers**:
  - Injected on both non-streaming and streaming completions (`/v1/chat/completions` and `/v1/responses`):
    - `X-AcademicAI-Request-Cost`
    - `X-AcademicAI-Input-Cost`
    - `X-AcademicAI-Output-Cost`
    - `X-AcademicAI-Prompt-Tokens`
    - `X-AcademicAI-Completion-Tokens`
    - `X-AcademicAI-Cost-Currency` (EUR)
    - `X-AcademicAI-Cost-Estimated`
  - Zero disruption to standard OpenAI payloads (no proprietary JSON fields injected).
- **Local Persistent Aggregator & Ring Buffer (`academicai/local_cost_tracker.py`)**:
  - Exactly-once request booking for non-streaming and streaming responses.
  - Aggregations across `all_time`, `today` (UTC YYYY-MM-DD), `this_month` (UTC YYYY-MM), `by_model`, and `by_client` (anonymized hash `client_<sha256[:8]>`).
  - Ring buffer of the last 500 requests (`ACADEMICAI_LOCAL_COST_HISTORY_LIMIT`) storing strictly accounting metadata (guaranteed: NO prompts, completions, tools, or API keys).
  - Atomic JSON persistence via `tempfile` + `os.replace` with Windows retry logic (`data/local_cost_cache.json`).
- **Extended Status Endpoint (`GET /internal/cost-status`)**:
  - Preserves 100% backward-compatible root fields for existing monitoring scripts.
  - Exposes `backend_cost_monitoring` snapshot and comprehensive `local_cost_tracking` details with aggregations, pricing cache health, and recent request history.
- **Comprehensive Unit & Integration Test Suite**:
  - Added `tests/test_cost_calculation_unit.py` (17 tests) and `tests/test_local_cost_tracker_unit.py` (7 tests).
  - Expanded `tests/test_cost_headers.py` and `tests/test_characterization_endpoints.py` with local cost verification.
  - Total test suite expanded to 233 passing tests.

## 0.7.1 - 2026-09-12

### ⚙️ Multi-Model Compatibility & Token Limit Auto-Stripping
- **Gemini & O3 Token Limit Auto-Stripping**: Automatically strip `max_tokens` / `max_completion_tokens` for Google Gemini models (`gemini-*`) and OpenAI `o3` in `academicai/transformation.py` to bypass upstream HTTP 500 errors (`internalErrorCode: 605`).
- **Reasoning Model Alignment**: Refined `reasoningEffort` injection when `temperature` is present to target only models that mandate it (`claude-opus-4-6`, `gpt-5.2`, `gpt-5.5`), ensuring `claude-opus-4-8` and `o3` function without validation errors.
- **100% Model Connectivity Verification**: All 16 active models in the AcademicAI catalog (`claude-opus-4-8`, `gpt-5.5`, `o3`, `gemini-3.5-flash`, etc.) verified working across both non-streaming and streaming calls via `test_models_connectivity.py`.
- **Unit Tests**: Added automated tests for parameter filtering and token limit auto-stripping in `tests/test_transformation_sticky_system.py`.

## 0.7 - 2026-09-11

### 🤖 OpenAI Codex & Responses API
- **`POST /v1/responses`**: Added fully compliant OpenAI Responses API endpoint for OpenAI Codex CLI and Codex Desktop App (`wire_api = "responses"`).
- **Shared Canonical Engine**: Zero code duplication — both `/v1/chat/completions` and `/v1/responses` invoke `_execute_completion_pipeline`.
- **Codex Wire Protocol Support**: Normalizes top-level `instructions`, structured `input` arrays (`message`, `function_call`, `function_call_output`), and flat tool parameter schemas.
- **SSE Streaming Events**: Implements OpenAI Responses streaming events (`response.created`, `response.in_progress`, delta text/function_call chunks, `response.output_item.done`, `response.completed`).
- **Token Accounting**: Emits `input_tokens` and `output_tokens` (strictly required by Codex CLI's Rust deserializer) alongside standard `prompt_tokens` and `completion_tokens`.
- **Multi-Turn Tool Roundtrips**: Emits tool calls to Codex for client-side local sandbox execution, receiving `function_call_output` in follow-up turns.
- **Codex Documentation**: Added [`docs/codex.md`](docs/codex.md) with comprehensive configuration guide (`config.toml`) and usage instructions.

### 🏗️ Domain Modularization & Modernization
- **Modular Domain Architecture**: Decomposed monolithic `server.py` into cohesive domain modules:
  - `academicai/config.py`: Typed configuration and safe import-time validation.
  - `academicai/request_guards.py`: Request body validation, schema enforcement, and rate limiting with TTL bucket sweep.
  - `academicai/cost_monitoring.py`: Cost API cache, atomic file operations, and status computation.
  - `academicai/runtime.py`: PID file lifecycle and backend health checks.
  - `academicai/logging_config.py`: Windows-safe rotating log handlers and Uvicorn logger wiring.
  - `academicai/humanization.py`: Target channel detection and 2nd-pass humanization.
  - `academicai/app.py`: FastAPI app factory and modern ASGI `lifespan` context manager (deprecating legacy `@app.on_event`).
  - `academicai/responses.py`: Responses API request normalization and SSE event builder.
  - `server.py`: Slim compatibility entrypoint and CLI runner (< 200 lines).
- **Tool Emulation Upgrades**:
  - High-density TypeScript tool signatures (`_compact_tool_def`) with concise enum unions, typed arrays, defaults, and shallow objects.
  - Hard `tool_choice` enforcement (`required`, specific target, or `none`).
  - JSON-repair sanitization (trailing comma removal, lenient control chars, Windows path escape handling).
  - Standardized XML observation tags (`<tool_result id="..." name="...">`).
- **Windows Reliability**: Implemented `SafeTimedRotatingFileHandler` preventing `WinError 32` file-lock conflicts during log rotation.

### 🧪 Test Suite & Safety Baselines
- Expanded test coverage from 19 to 186 passing offline tests.
- Added comprehensive unit and integration suites: `test_responses_api.py`, `test_app_unit.py`, `test_logging_config_unit.py`, `test_runtime_unit.py`, `test_cost_monitoring_unit.py`, `test_request_guards.py`, `test_config.py`, `test_tool_emulation_unit.py`, `test_characterization_endpoints.py`.
- Test discovery safety baseline (`pytest.ini` scoped to `tests/`) and isolated test port fallback (`11436`).

## 0.6 - 2026-07-17

### 🔒 Security & Hardening
- Mandatory API key validation: length checks and placeholder detection prevent misconfigured keys from reaching the backend.
- Request size and schema validation for `/v1/chat/completions` — malformed payloads are rejected early.
- Rate limiting added for chat requests to protect the backend under load.
- New `/health` endpoint reports backend reachability and basic proxy status.
- Debug dumps now redact sensitive fields (API keys, tokens) before logging.
- Added dedicated security hardening test suite (`tests/test_hardening_security_runtime.py`).

### ⚙️ Non-OpenAI Model Compatibility
- `response_format` is now stripped for non-OpenAI models to bypass a backend validation bug.
- `max_tokens` / `max_completion_tokens` are auto-stripped for gpt-5 and Perplexity models to prevent upstream 500 errors.
- `frequency_penalty` and `presence_penalty` are filtered out for non-OpenAI models.
- `reasoningEffort` is forced to `none` for all reasoning models whenever `temperature` is present, satisfying backend validation for gpt-5.2, gpt-5.5, and their variants.

### 🛠️ Operations & Infrastructure
- **Log rotation**: Daily `TimedRotatingFileHandler` with 30-day retention — logs no longer grow unbounded.
- **Dynamic port configuration**: `ACADEMICAI_PROXY_PORT` env var is now respected by `start_server.ps1` and `stop_server.ps1`, removing the hardcoded default.
- **Isolated test port**: Local test environment uses port 11436 (`env.localtest`) to prevent conflicts with a running production server.
- `env.localtest` moved to `.gitignore` — local test credentials stay out of version control.

### 🧪 Testing
- Full test suite refactored to `httpx` + `pytest` for cleaner async integration tests.
- Local test environment now imports backend credentials from `.env` automatically.
- New `test_models_connectivity.py` script verifies connectivity to all configured AcademicAI models on demand.
- Background test server PID tracking fixed; tests prefer the local venv Python.

### 📬 Skill Snippets
- Added `mail-processor` shadow snippet for Copilot-driven mail workflows.
- Skill snippet file brought back into sync with latest snippet schema.

### 📝 Documentation
- README expanded with AcademicAI Cost API constraints, isolated port guidance, log rotation notes, and Copilot-centric Skill Snippet advice.
- Corrected default value of `ACADEMICAI_ENABLE_SKILL_SNIPPETS` (was incorrectly documented as `true`, default is `false`).

## 0.2.4 - 2026-03-06
- Cost monitoring switched to explicit opt-in via `ACADEMICAI_ENABLE_COST_MONITORING` (default `false`).
- When disabled, proxy performs no automatic calls to AcademicAI `/api/v1/cost` and emits no cost headers.
- Cost status/header schema aligned to API docs (`totalCost`, `totalClients`, `costs[]` count) instead of inferred `remaining_credit`.
- Added/updated tests:
  - `tests/test_cost_headers.py` (doc-aligned cost headers)
  - `tests/test_cost_monitoring_toggle.py` (headers absent when monitoring disabled)
- Test suite files moved from repository root into `tests/` for cleaner project structure.
- `.env.example` extended with cost-monitoring flags and default-off guidance.
- README updated with a dedicated cost-monitoring section, explicit "currently untested" note, and required AcademicAI permission (`ACCESS_API_MONITOR_CREDIT`).

## 0.2.3 - 2026-03-05
- Public release hardening and tenant-neutralization pass.
- Removed tenant-specific/BOKU references from tracked code, docs, snippets, and tests.
- `.env.example` made generic (`ACADEMICAI_BASE_URL` placeholder), added `ACADEMICAI_PROXY_API_KEY` and `TENANT_ID` examples.
- Added agent-friendly installation guidance to `README.md` (recommended GitHub-URL handoff workflow + verification checklist).
- Added tenant templates under `docs/tenant-template/` (`env.template`, `skill_snippets.template.json`).
- Added `pytest` to `requirements.txt` so test execution is part of standard installation flow.
- Security logging fix: removed clear-text API key logging at startup (CodeQL finding remediation).
- Test suite re-run after changes: `19 passed`.

## 0.1.2 - 2026-02-28
- Added optional self-learning updates for `skill_snippets.json` (variant 1, keyword-based).
  - New env flags:
    - `ACADEMICAI_ENABLE_AUTO_SKILL_LEARNING`
    - `ACADEMICAI_AUTO_SKILL_TOPICS_PER_CALL`
    - `ACADEMICAI_AUTO_SKILL_MIN_TOPIC_LEN`
  - Proxy now upserts `auto:<tool_name>` snippets from successful tool-call outputs.
  - Auto snippets track `hits` and `last_updated` and extend topics over time.
- Added tests for auto-learning creation and update behavior in `test_skill_snippets.py`.
- README updated with planned/open feature list and self-learning docs.
- Mail-move guard widened: `message move` is now guarded for all target folders (not only Cabinet).
- Skill snippet guidance updated with correct Himalaya move/copy argument order (`<TARGET> <ID>...`).

## 0.1.1 - 2026-02-27
- Multi-step tool-call emulation added:
  - JSON schema now supports `action: "tool_calls"` with `calls[]`.
  - Backward compatible with legacy single-call `action: "tool_call"`.
- Response builders now support multiple tool calls (non-stream + SSE stream deltas).
- Server switched to batch parsing (`parse_tool_calls`) and multi-call forwarding.
- Safety guard added for mail workflows: blocks `exec`-based `message delete/move ... Cabinet`
  when no prior `write/edit` exists in the same call batch.
- If an unsafe delete/move is blocked and no safe tool call remains, proxy returns
  a clear plain-text explanation instead of raw JSON.
- New tests:
  - `test_multi_step_tool_emulation.py`
  - extended `test_post_tool_guard.py` for write-before-delete enforcement.

## 0.1.0 - 2026-02-27
- Initial public-light release draft.
- OpenAI-compatible FastAPI proxy for AcademicAI.
- Tool-call emulation via JSON-mode.
- Human-target JSON fallback formatting improvements.
- Model/style default split for chat vs tool-mode.
- Basic test procedure documented.
