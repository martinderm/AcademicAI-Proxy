"""
AcademicAI OpenAI-kompatibler Proxy Server

Exponiert AcademicAI als OpenAI-kompatible API auf Port 11435.
OpenClaw (und andere Tools) können ihn wie jeden OpenAI-kompatiblen Provider nutzen.

Start:
    py server.py

Endpoints:
    GET  /v1/models                 → Modell-Liste
    POST /v1/chat/completions       → Chat Completion (inkl. Streaming-Emulation)
    GET  /health                    → Health Check
"""

import os
import sys
import time
import uuid
import json
import logging

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.concurrency import run_in_threadpool
import uvicorn

import academicai
from academicai.tool_emulation import (
    inject_tools_into_messages,
    parse_tool_call,
    extract_respond_content,
    format_arbitrary_json_as_codeblock,
    format_arbitrary_json_for_humans,
    build_tool_calls_response,
    build_tool_calls_sse_chunks,
)


def _extract_text_content(msg_content) -> str:
    """Normalisiert OpenAI-Message-Content zu Plain-Text."""
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        parts = []
        for item in msg_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)
    return ""


def _is_human_readable_target(messages: list) -> bool:
    """
    Heuristik: Nur bei menschlichen Zielkanälen JSON->Human-Text-Fallback aktivieren.

    False für klar maschinelle Runs (z.B. cron).
    True für typische Human-Channels (whatsapp/telegram/signal/discord/slack/webchat...).
    """
    user_text = "\n".join(
        _extract_text_content(m.get("content"))
        for m in messages
        if m.get("role") == "user"
    ).lower()

    # Explizit maschineller Trigger
    if "[cron:" in user_text:
        return False

    # Chat-Metadaten aus OpenClaw-User-Envelope (auch ohne system channel marker)
    user_human_markers = [
        "conversation info (untrusted metadata)",
        '"is_group_chat": true',
        '"is_group_chat": false',
        '"conversation_label":',
        '"sender": "+',
    ]
    if any(marker in user_text for marker in user_human_markers):
        return True

    system_text = "\n".join(
        _extract_text_content(m.get("content"))
        for m in messages
        if m.get("role") == "system"
    ).lower()

    human_channel_markers = [
        "channel=whatsapp", '"channel": "whatsapp"',
        "channel=telegram", '"channel": "telegram"',
        "channel=signal", '"channel": "signal"',
        "channel=imessage", '"channel": "imessage"',
        "channel=discord", '"channel": "discord"',
        "channel=slack", '"channel": "slack"',
        "channel=googlechat", '"channel": "googlechat"',
        "channel=irc", '"channel": "irc"',
        "channel=webchat", '"channel": "webchat"',
        '"chat_type": "group"', '"chat_type": "direct"',
    ]
    if any(marker in system_text for marker in human_channel_markers):
        return True

    # OpenClaw-Session ohne explizite Channel-Marker -> für Nutzer standardmäßig als human behandeln
    if "you are a personal assistant running inside openclaw." in system_text:
        return True

    # Sonst eher API-/Maschinenverkehr
    return False

# --- Config ---
PORT = int(os.environ.get("ACADEMICAI_PROXY_PORT", 11435))
API_KEY = os.environ.get("ACADEMICAI_PROXY_API_KEY", "academicai-proxy")
DEBUG_DUMPS = os.environ.get("ACADEMICAI_DEBUG_DUMPS", "false").lower() in ("1", "true", "yes", "on")

# Proxy-Defaults (nur wenn Client keinen Wert setzt)
DEFAULT_CHAT_TEMPERATURE = float(os.environ.get("ACADEMICAI_DEFAULT_CHAT_TEMPERATURE", "0.6"))
DEFAULT_TOOL_TEMPERATURE = float(os.environ.get("ACADEMICAI_DEFAULT_TOOL_TEMPERATURE", "0.1"))
DEFAULT_CHAT_VERBOSITY = os.environ.get("ACADEMICAI_DEFAULT_CHAT_VERBOSITY", "medium")
DEFAULT_TOOL_VERBOSITY = os.environ.get("ACADEMICAI_DEFAULT_TOOL_VERBOSITY", "low")
DEFAULT_TOOL_REASONING_EFFORT = os.environ.get("ACADEMICAI_DEFAULT_TOOL_REASONING_EFFORT", "low")

# --- Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("academicai-proxy")

app = FastAPI(
    title="AcademicAI Proxy",
    description="OpenAI-kompatibler Proxy für AcademicAI (BOKU)",
    version="1.0.0",
)

security = HTTPBearer(auto_error=False)


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials is None or credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.credentials


# --- Health ---

@app.get("/health")
def health():
    return {"status": "ok", "service": "academicai-proxy"}


# --- Models ---

@app.get("/v1/models")
async def list_models(key: str = Depends(verify_key)):
    try:
        return await run_in_threadpool(academicai.get_models)
    except Exception as e:
        log.error(f"get_models failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))


# --- Chat Completions ---

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, key: str = Depends(verify_key)):
    body = await request.json()

    # Vollständiges Request-Dump für Debugging (optional via Env)
    if DEBUG_DUMPS:
        import json as _json
        _dump_path = os.path.join(os.path.dirname(__file__), "last_request.json")
        try:
            with open(_dump_path, "w", encoding="utf-8") as _f:
                _json.dump(body, _f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    model = body.get("model")
    messages = list(body.get("messages") or [])

    # Top-level "system" Parameter (z.B. von Anthropic-style Clients) → in messages einfügen
    top_level_system = body.get("system")
    if top_level_system and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": top_level_system})

    # Tools extrahieren — werden via Prompt-Injection emuliert
    tools = body.get("tools") or body.get("functions") or []
    has_tools = bool(tools)
    if ("tool_choice" in body) and not has_tools:
        log.warning("tool_choice provided without tools; ignoring tool emulation for this request")
    log.info(f"incoming: model={model} stream={body.get('stream')} roles={[m.get('role') for m in messages]} tools={len(tools)} has_tools={has_tools}")

    # Tool-Definitionen in System-Prompt injizieren
    if tools:
        messages = inject_tools_into_messages(messages, tools)

    if not model or not messages:
        raise HTTPException(status_code=422, detail="model und messages sind Pflichtfelder")

    want_stream = bool(body.get("stream"))
    human_target_hint = _is_human_readable_target(messages)

    # Optionale Parameter weiterreichen — nur bekannte, AcademicAI-sichere Felder
    # tools / tool_choice / functions werden via Prompt-Injection emuliert (nicht nativ weitergegeben)
    optional = {}
    for field in [
        "temperature", "max_tokens", "max_completion_tokens",
        "frequency_penalty", "presence_penalty",
        "reasoning_effort", "verbosity", "seed", "stop",
    ]:
        if field in body:
            optional[field] = body[field]

    # Sinnvolle Proxy-Defaults (nur falls Client nichts gesetzt hat)
    if has_tools:
        optional.setdefault("temperature", DEFAULT_TOOL_TEMPERATURE)
        # GPT-5-Modelle profitieren bei Emulation von knapper, deterministischerem Stil
        if model and "gpt-5" in model:
            optional.setdefault("verbosity", DEFAULT_TOOL_VERBOSITY)
            optional.setdefault("reasoning_effort", DEFAULT_TOOL_REASONING_EFFORT)
    elif human_target_hint:
        optional.setdefault("temperature", DEFAULT_CHAT_TEMPERATURE)
        if model and "gpt-5" in model:
            optional.setdefault("verbosity", DEFAULT_CHAT_VERBOSITY)

    # response_format: bei Tool-Emulation JSON-Mode erzwingen,
    # sonst Wert aus Request durchreichen (ausser json_schema)
    if tools:
        optional["response_format"] = {"type": "json_object"}
    elif "response_format" in body:
        rf = body.get("response_format")
        if isinstance(rf, dict):
            if rf.get("type") != "json_schema":
                optional["response_format"] = rf
        else:
            log.warning("ignoring non-dict response_format from client")

    # tailoredAiId via extra_body
    if "extra_body" in body and "tailoredAiId" in body["extra_body"]:
        optional["extra_body"] = {"tailoredAiId": body["extra_body"]["tailoredAiId"]}

    try:
        response = await run_in_threadpool(academicai.completion, model=model, messages=messages, **optional)
    except Exception as e:
        log.error(f"completion failed: model={model} error={e}")
        raise HTTPException(status_code=502, detail=str(e))

    completion_id = response.id
    created_ts = response.created
    resp_model = response.model
    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = choice.finish_reason or "stop"
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    # JSON-Mode Response verarbeiten (nur wenn Tools im Request waren)
    tool_call_data = None
    human_target = human_target_hint
    if has_tools:
        tool_call_data = parse_tool_call(content)
        if tool_call_data:
            log.info(f"tool_call detected: {tool_call_data['name']}({list(tool_call_data.get('arguments', {}).keys())})")
        else:
            # Kein Tool-Call — entweder {"action":"respond",...} oder Fallback
            extracted = extract_respond_content(content)
            if extracted is not None:
                log.info(f"json_mode respond: content_len={len(extracted)}")
                content = extracted

                # GPT-5 liefert teils action=respond mit JSON-String in content.
                # Auf Human-Targets trotzdem in natürlich lesbaren Text umformen.
                if human_target:
                    humanized_from_content = format_arbitrary_json_for_humans(content)
                    if humanized_from_content is not None:
                        log.warning("json_mode respond: JSON-string content -> human text (human target)")
                        content = humanized_from_content
            else:
                # Letzter Fallback nur für human-readable Targets.
                # Für maschinelle Flows (z.B. cron) bleibt raw content erhalten.
                if human_target:
                    human_text = format_arbitrary_json_for_humans(content)
                    if human_text is not None:
                        log.warning(f"json_mode: arbitrary JSON -> human text (human target): {content[:80]}")
                        content = human_text
                    else:
                        # Fallback-Fallback: falls Rendern scheitert, wenigstens lesbar
                        codeblock = format_arbitrary_json_as_codeblock(content)
                        if codeblock is not None:
                            log.warning(f"json_mode: arbitrary JSON -> code block fallback (human target): {content[:80]}")
                            content = codeblock
                        else:
                            log.warning(f"json_mode parse failed, using raw content: {content[:120]}")
                else:
                    log.warning("json_mode: arbitrary JSON on non-human target, keeping raw content")

    # Wenn Streaming gewünscht: Antwort als SSE emulieren
    if want_stream:
        def sse_generator():
            if tool_call_data:
                # Tool-Call-Chunks im OpenAI-Streaming-Format
                for chunk in build_tool_calls_sse_chunks(completion_id, created_ts, resp_model, tool_call_data):
                    yield f"data: {json.dumps(chunk)}\n\n"
            else:
                # Normaler Text-Response als SSE
                # Chunk 1: role delta
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': resp_model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]})}\n\n"
                # Chunk 2: Content
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': resp_model, 'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}]})}\n\n"
                # Chunk 3: finish
                yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created_ts, 'model': resp_model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # Kein Streaming: normaler JSON-Response
    if tool_call_data:
        return build_tool_calls_response(completion_id, created_ts, resp_model, tool_call_data, usage)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": resp_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


# --- Start ---

if __name__ == "__main__":
    log.info(f"AcademicAI Proxy startet auf Port {PORT}")
    log.info(f"API-Key: {API_KEY}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
