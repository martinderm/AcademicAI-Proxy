# Changelog

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
