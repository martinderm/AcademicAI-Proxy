# Architecture Refactoring & Modularization Plan

## 1. Executive Summary

This document records the modularization and architectural hardening of `AcademicAI-Proxy`. The monolithic legacy entrypoint `server.py` (~1150 lines) was systematically decomposed into single-responsibility domain modules under [`academicai/`](../../academicai/) following the Daedalus 7-phase engineering standard (**"Real Engineering, not Vibe Coding"**, TDD-first, Single-Harness Locking, relative paths only).

---

## 2. Refactoring Objectives & Architectural Invariants

1. **Single Responsibility Principle (SRP):** Decompose monolithic server logic into cohesive, independently testable domain modules.
2. **Backward Compatibility:** All existing entrypoints, test monkeypatches (`server.MAX_MESSAGES`, `server.academicai.completion`, `server._check_backend_health` etc.), and external harness contracts remain 100% functional without test changes.
3. **Lifespan Modernization:** Replace deprecated FastAPI `@app.on_event("startup")` and `@app.on_event("shutdown")` with modern ASGI `@asynccontextmanager async def lifespan(app: FastAPI)`.
4. **Resilience & Safety:** Inbound size validation (413/422), token-bucket rate limiting (429) with TTL memory leak sweep, Windows-safe atomic cache writes and file handler cleanup.
5. **Tool Emulation Modernization:** High-density TypeScript signatures, strict `tool_choice` enforcement (`"required"`, specific function), JSON-repair sanitization, and `<tool_result>` XML observation tags. Core tool-emulation heuristics were preserved.
6. **Path Discipline:** Purely repository-relative paths across code, configuration, and documentation (public GitHub repository compliance).

---

## 3. Module Breakdown & Domain Responsibilities

| Module | Primary Responsibility | Key Functions / Classes |
| :--- | :--- | :--- |
| [`academicai/config.py`](../../academicai/config.py) | Typed configuration and environment parsing | `Settings`, `get_settings()`, `validate_config()`, `_validate_proxy_api_key()` |
| [`academicai/request_guards.py`](../../academicai/request_guards.py) | Inbound payload validation & rate limiting | `validate_request_json_size()`, `validate_chat_request_body()`, `enforce_chat_rate_limit()`, `prune_rate_limit_buckets()` |
| [`academicai/transformation.py`](../../academicai/transformation.py) | Message normalization & text extraction SSOT | `extract_text_content()`, `_normalize_messages()`, Azure prefix caching |
| [`academicai/tool_emulation.py`](../../academicai/tool_emulation.py) | Tool prompt injection, call parsing & repair | `inject_tools_into_messages()`, `parse_tool_calls()`, `apply_post_tool_guard()`, `build_tool_calls_response()` |
| [`academicai/cost_monitoring.py`](../../academicai/cost_monitoring.py) | BOKU cost cache & header injection | `get_cost_cache_with_lazy_refresh()`, `build_cost_headers()`, `write_cost_cache()`, `get_cost_status_payload()` |
| [`academicai/runtime.py`](../../academicai/runtime.py) | PID file management & backend health checks | `write_pid_file()`, `cleanup_pid_file()`, `check_backend_health()`, `get_health_payload()` |
| [`academicai/logging_config.py`](../../academicai/logging_config.py) | Log rotation & uvicorn logger wiring | `configure_logging()`, `close_handlers()`, `log` |
| [`academicai/humanization.py`](../../academicai/humanization.py) | Target channel detection & response smoothing | `is_human_readable_target()`, `last_user_text()`, `build_humanization_messages()`, `run_humanization_pass()` |
| [`academicai/app.py`](../../academicai/app.py) | FastAPI application factory & lifespan | `create_app()`, `app`, `lifespan`, route handlers (`/health`, `/internal/cost-status`, `/v1/models`, `/v1/chat/completions`) |
| [`server.py`](../../server.py) | Thin CLI runner & compatibility layer | CLI startup banner, `uvicorn.run()`, legacy symbol re-exports (< 200 lines) |

---

## 4. Implemented Ticket Traceability

- **T01 (Safety Baseline):** Established testpaths in `pytest.ini`, test port isolation (`11436`), and characterization test suite (`tests/test_characterization_endpoints.py`).
- **T02 (Typed Config):** Extracted `academicai/config.py` with `Settings` dataclass and non-crashing import behavior.
- **T03 (Request Guards):** Extracted `academicai/request_guards.py` with 413/422 guards and TTL sweep for rate limit buckets.
- **T04 (Pure Purge):** Removed obsolete skill snippets, harness-specific mail-delete guard, and hardcoded PowerShell prompt heuristics.
- **T04b (Emulation Modernization):** Upgraded tool signatures to concise TypeScript notation, added `tool_choice` enforcement, JSON repair, and `<tool_result>` tags.
- **T05 (Cost Monitoring):** Extracted `academicai/cost_monitoring.py` with Windows-safe atomic cache replacement and lazy background refreshing.
- **T06 (Runtime Lifecycle):** Extracted `academicai/runtime.py` with PID file verification and backend health diagnostics.
- **T07 (Logging Setup):** Extracted `academicai/logging_config.py` with daily rotating handlers and Windows handle cleanup.
- **T08 (Humanization Flow):** Extracted `academicai/humanization.py` with messenger channel heuristics and second-pass LLM rewriting.
- **T09 (App Factory & Lifespan):** Created `academicai/app.py` with modern ASGI `lifespan` handler and route wiring.
- **T10 (Entrypoint Contracting):** Contracted `server.py` into a thin CLI runner and compatibility facade with acyclic import verification.
- **T11 (Documentation & Hygiene):** Moved concept and archive documents to `docs/architecture/` and `docs/archive/`, cleaned root directory logs, and updated System Map.
