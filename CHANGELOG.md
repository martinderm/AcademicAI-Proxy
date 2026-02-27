# Changelog

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
