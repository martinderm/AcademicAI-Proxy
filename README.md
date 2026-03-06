# AcademicAI Proxy

OpenAI-compatible proxy for AcademicAI.
It exposes AcademicAI models on a local OpenAI-style API (default: `http://127.0.0.1:11435/v1`).

## Status

- Chat completions: ✅
- Model list endpoint: ✅
- Health endpoint: ✅
- Tool-call emulation (JSON-mode): ✅
- SSE-style streaming emulation: ✅

### Planned / open features

- Skill snippets on vector basis: 🔴
- TailoredAI RAG interface (per API docs, short re-check): 🔴
- Cost/usage monitoring via API cost endpoint: 🔴

## Why this proxy exists

AcademicAI does not provide native OpenAI function-calling/tool-calling in the same way OpenAI-compatible clients expect.
This proxy emulates the tool flow so orchestrators (e.g. OpenClaw) can still run tools reliably.

### Important: limits of tool-call emulation (plain language)

Tool-calling is **simulated**, not native. That means the model is guided via prompt + JSON parsing,
not by a backend-level function-calling engine.

In practice this works well, but there are limits:
- behavior is probabilistic (occasionally the model may answer in JSON style instead of ideal natural text)
- extra guardrails are needed to avoid unnecessary repeated tool calls
- reliability is generally lower than true native tool-calling APIs

So: good for practical use, but not mathematically deterministic.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Authentication

Client -> Proxy (Bearer):
- `Authorization: Bearer <YOUR_PROXY_API_KEY>`

Proxy -> AcademicAI backend:
- `X-Client-ID: <ACADEMICAI_CLIENT_ID>`
- `X-Client-Secret: <ACADEMICAI_CLIENT_SECRET>`

Configure these values in `.env` (never commit real secrets).

## Tenant separation (generic repo vs external tenant config)

Keep repository content generic. Put tenant-specific values outside the repository:
- `.env` with endpoint, client ID/secret, proxy API key
- optional tenant snippets file via `ACADEMICAI_SKILL_SNIPPETS_FILE`

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
```

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

### Optional skill snippet injection (tool-call reliability)

You can enable retrieval-based skill snippets that are injected as short system context
before tool-emulation. This increases the chance of correct tool use for domain intents
(e.g. mailbox/email -> Himalaya wrapper commands).

Env flags:
- `ACADEMICAI_ENABLE_SKILL_SNIPPETS=true|false`
- `ACADEMICAI_SKILL_SNIPPETS_FILE=./skill_snippets.json`
- `ACADEMICAI_SKILL_SNIPPETS_MAX=1`

Notes:
- Injection runs only in tool mode (`tools`/`functions` present).
- Snippets are selected by topic match against the latest user message.

### Optional self-learning snippet updates (variant 1, keyword-based)

The proxy can auto-update `skill_snippets.json` from successful tool-call decisions.
This is intentionally simple (no vector DB / no embeddings):
- derive keywords from the latest user request
- upsert `auto:<tool_name>` snippets
- increase hit counters and extend topics over time

Env flags:
- `ACADEMICAI_ENABLE_AUTO_SKILL_LEARNING=true|false`
- `ACADEMICAI_AUTO_SKILL_TOPICS_PER_CALL=6`
- `ACADEMICAI_AUTO_SKILL_MIN_TOPIC_LEN=4`

Notes:
- Learning runs only when tool mode is active and a tool call was actually emitted.
- Existing manual snippets are preserved; auto snippets are marked with `source: "auto"`.

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

### Mail-delete safety guard (write-before-delete)

For mailbox workflows, the proxy enforces this batch rule:
- `exec` calls containing `message delete` or any `message move ...`
  are allowed only if a prior `write` or `edit` call exists in the same
  tool-call batch.

If such a delete/move is blocked and no safe tool call remains, the proxy returns
plain text:
- `Blocked unsafe mail action: message delete/move requires a prior write/edit in the same tool-call batch.`

## Tests

Run smoke + functional tests:

```powershell
py -m pytest -q
py test_tool_emulation.py
py -m pytest -q test_post_tool_guard.py
```

Optional:

```powershell
py test_openclaw_style.py
```

## Project layout

```
academicai-proxy/
  academicai/
    __init__.py
    auth.py
    errors.py
    provider.py
    tool_emulation.py
    transformation.py
  server.py
  start_server.ps1
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
