# AcademicAI Proxy

OpenAI-compatible proxy for AcademicAI.
It exposes AcademicAI models on a local OpenAI-style API (default: `http://127.0.0.1:11435/v1`).

## Status

- Chat completions: ✅
- Model list endpoint: ✅
- Health endpoint: ✅
- Tool-call emulation (JSON-mode): ✅
- SSE-style streaming emulation: ✅

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
   - tool call response (`finish_reason: tool_calls`)
   - normal assistant text
5. Upstream orchestrator executes tool and sends `role=tool` follow-up

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

## License

MIT (see `LICENSE`).
