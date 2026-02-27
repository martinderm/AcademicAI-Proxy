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
from pathlib import Path
from typing import Optional

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
    parse_tool_calls,
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


def _last_user_text(messages: list) -> str:
    """Liefert den letzten User-Text aus den Original-Messages."""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return _extract_text_content(m.get("content"))
    return ""


def _apply_post_tool_guard(messages: list, has_tools: bool) -> list:
    """
    Stabilisiert den Follow-up-Schritt nach einem Tool-Result.
    - Erfolgreiches Tool-Result: finale Antwort bevorzugen.
    - Fehlerhaftes Tool-Result: Erfolg NICHT behaupten, sondern korrigierten
      Tool-Call auslösen oder Fehler transparent melden.
    """
    if not has_tools or not messages:
        return messages

    last = messages[-1] or {}
    if last.get("role") != "tool":
        return messages

    tool_text = _extract_text_content(last.get("content")).lower()
    has_error = any(tok in tool_text for tok in ["error:", "cannot parse", "failed", "not found", "exception"])

    if has_error:
        guard_text = (
            "TOOL_RESULT_ERROR: The latest tool result contains an error. "
            "Do NOT claim success. Either issue a corrected tool_call, or explain the failure clearly. "
            "For Himalaya envelope search, keep options before query, e.g. envelope list -s 50 \"from wordpress@usage-ng.boku.ac.at\"."
        )
    else:
        guard_text = (
            "NO_FURTHER_TOOL_CALLS: You already received tool results. "
            "Now produce the final user-facing answer. "
            "Call another tool only if the latest tool result is clearly missing required data."
        )

    return [{"role": "system", "content": guard_text}] + messages


def _score_topic_match(user_text: str, topics: list) -> int:
    txt = (user_text or "").lower()
    score = 0
    for t in (topics or []):
        tok = str(t).strip().lower()
        if tok and tok in txt:
            score += 1
    return score


def _load_skill_snippets() -> list:
    try:
        p = Path(SKILL_SNIPPETS_FILE)
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        log.warning(f"skill snippets load failed: {e}")
        return []


def _inject_skill_snippet_context(messages: list, user_text: str) -> list:
    """Injiziert passende Skill-Snippets als kurze System-Message."""
    if not ENABLE_SKILL_SNIPPETS:
        return messages

    snippets = _load_skill_snippets()
    if not snippets:
        return messages

    scored = []
    for s in snippets:
        score = _score_topic_match(user_text, s.get("topics", []))
        if score > 0 and s.get("snippet"):
            scored.append((score, s))

    if not scored:
        return messages

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [s for _, s in scored[: max(1, SKILL_SNIPPETS_MAX)]]
    selected_ids = [str(s.get("id", "snippet")) for s in selected]
    log.info(f"skill snippet injection: selected_ids={selected_ids}")

    parts = [
        "SKILL CONTEXT (retrieved): Use this operational guidance when deciding tool calls."
    ]
    for s in selected:
        sid = s.get("id", "snippet")
        parts.append(f"[{sid}] {s.get('snippet', '').strip()}")

    msg = {"role": "system", "content": "\n\n".join(parts)}
    return [msg] + messages


def _is_mail_delete_exec_call(call: dict) -> bool:
    """Erkennt exec-Calls, die Himalaya-Mails löschen/verschieben."""
    if not isinstance(call, dict) or call.get("name") != "exec":
        return False
    args = call.get("arguments") or {}
    cmd = str(args.get("command", "")).lower()
    return ("message delete" in cmd) or ("message move" in cmd and "cabinet" in cmd)


def _enforce_write_before_mail_delete(tool_calls: list[dict]) -> tuple[list[dict], bool]:
    """
    Safety-Guard für Batch-Tool-Calls:
    Mail-Delete/Move darf in derselben Batch nur passieren, wenn vorher ein write/edit Call enthalten ist.

    Returns: (filtered_calls, blocked_any)
    """
    if not tool_calls:
        return [], False

    out = []
    blocked_any = False
    has_write_before = False
    for c in tool_calls:
        name = (c or {}).get("name")
        if name in ("write", "edit"):
            has_write_before = True
            out.append(c)
            continue

        if _is_mail_delete_exec_call(c) and not has_write_before:
            blocked_any = True
            log.warning("blocked unsafe mail delete/move call without prior write/edit in same batch")
            continue

        out.append(c)

    return out, blocked_any


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

# Optionaler zweiter Pass: strukturiertes Ergebnis -> natürlichsprachliche Endantwort
ENABLE_HUMANIZATION_PASS = os.environ.get("ACADEMICAI_ENABLE_HUMANIZATION_PASS", "false").lower() in ("1", "true", "yes", "on")
HUMANIZATION_MODEL = os.environ.get("ACADEMICAI_HUMANIZATION_MODEL", "").strip()
HUMANIZATION_TEMPERATURE = float(os.environ.get("ACADEMICAI_HUMANIZATION_TEMPERATURE", "0.2"))

# Optional: skill snippet retrieval/injection to improve tool-call reliability
ENABLE_SKILL_SNIPPETS = os.environ.get("ACADEMICAI_ENABLE_SKILL_SNIPPETS", "false").lower() in ("1", "true", "yes", "on")
SKILL_SNIPPETS_FILE = os.environ.get("ACADEMICAI_SKILL_SNIPPETS_FILE", str(Path(__file__).with_name("skill_snippets.json")))
SKILL_SNIPPETS_MAX = int(os.environ.get("ACADEMICAI_SKILL_SNIPPETS_MAX", "1"))


def _build_humanization_messages(original_user_query: str, structured_content: str) -> list:
    """Prompt für den optionalen zweiten LLM-Pass (Humanisierung)."""
    system_msg = {
        "role": "system",
        "content": (
            "You rewrite structured tool output into a natural final answer for a human chat. "
            "Return only the final answer text for the user. "
            "Do NOT include JSON, code blocks, field names, metadata, or debug info."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Original user question:\n{original_user_query.strip() or '-'}\n\n"
            f"Structured/tool-derived result:\n{structured_content.strip()}\n\n"
            "Task: Write a concise, natural-language final reply for the user."
        ),
    }
    return [system_msg, user_msg]


async def _run_humanization_pass(model: str, original_user_query: str, structured_content: str) -> Optional[str]:
    """Führt optionalen zweiten LLM-Pass aus und liefert finalen Text zurück."""
    try:
        human_model = HUMANIZATION_MODEL or model
        resp = await run_in_threadpool(
            academicai.completion,
            model=human_model,
            messages=_build_humanization_messages(original_user_query, structured_content),
            temperature=HUMANIZATION_TEMPERATURE,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception as e:
        log.warning(f"humanization pass failed, fallback to first-pass content: {e}")
        return None

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

    original_user_query = _last_user_text(messages)

    # Tools extrahieren — werden via Prompt-Injection emuliert
    tools = body.get("tools") or body.get("functions") or []
    has_tools = bool(tools)
    if ("tool_choice" in body) and not has_tools:
        log.warning("tool_choice provided without tools; ignoring tool emulation for this request")
    log.info(f"incoming: model={model} stream={body.get('stream')} roles={[m.get('role') for m in messages]} tools={len(tools)} has_tools={has_tools}")

    # Optional: passende Skill-Snippets injizieren (z.B. mailbox/email -> Himalaya wrapper)
    if has_tools:
        messages = _inject_skill_snippet_context(messages, user_text=original_user_query)

    # Bei Follow-up nach Tool-Result finale Antwort stärker priorisieren
    messages = _apply_post_tool_guard(messages, has_tools=has_tools)

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
    tool_calls_data = []
    blocked_unsafe_delete = False
    human_target = human_target_hint
    if has_tools:
        tool_calls_data = parse_tool_calls(content)
        if tool_calls_data:
            tool_calls_data, blocked_unsafe_delete = _enforce_write_before_mail_delete(tool_calls_data)
            names = [c.get("name", "?") for c in tool_calls_data]
            log.info(f"tool_call(s) detected: count={len(tool_calls_data)} names={names}")
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

    # Klarer User-Text wenn ein unsicherer Delete-Call geblockt wurde
    if has_tools and blocked_unsafe_delete and not tool_calls_data:
        content = (
            "Blocked unsafe mail action: message delete/move requires a prior write/edit "
            "in the same tool-call batch."
        )
        finish_reason = "stop"

    # Optionaler zweiter Pass: natürliche Endantwort für Human-Channels
    if (
        ENABLE_HUMANIZATION_PASS
        and human_target
        and has_tools
        and not tool_calls_data
        and (content or "").strip()
    ):
        humanized = await _run_humanization_pass(model=resp_model, original_user_query=original_user_query, structured_content=content)
        if humanized:
            log.info(f"humanization pass applied: len_before={len(content)} len_after={len(humanized)}")
            content = humanized

    # Wenn Streaming gewünscht: Antwort als SSE emulieren
    if want_stream:
        def sse_generator():
            if tool_calls_data:
                # Tool-Call-Chunks im OpenAI-Streaming-Format
                for chunk in build_tool_calls_sse_chunks(completion_id, created_ts, resp_model, tool_calls_data):
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
    if tool_calls_data:
        return build_tool_calls_response(completion_id, created_ts, resp_model, tool_calls_data, usage)

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
