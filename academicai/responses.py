"""
OpenAI Responses API Domain Logic for AcademicAI-Proxy.

Provides:
- Inbound Responses API request normalization (`normalize_responses_request`)
  Mapping instructions, structured input items (messages, function_calls, function_call_outputs),
  and tools into canonical proxy format.
- Output serialization for non-streaming (`build_responses_output`)
- Standard OpenAI Responses SSE wire event generator (`build_responses_sse_events`)
"""

import json
import time
import uuid
from typing import Any, Iterator, Optional

from academicai.transformation import extract_text_content


def normalize_responses_request(
    body: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    """
    Normalisiert einen OpenAI Responses API Inbound Request in die kanonischen
    Proxy-Parameter (model, messages, tools, tool_choice, optional_params).

    Unterstützt:
    - instructions: wird als führende developer-Message normalisiert
    - input: String oder Liste von Items:
        * type="message" (oder Dict mit "role")
        * type="function_call" (Tool-Aufruf des Modells aus Vorrunden)
        * type="function_call_output" (Tool-Ergebnis vom Client/Codex)
    - tools: Liste von Tools (flach oder verschachtelt unter "function", Unpacking von "namespace")
    - tool_choice: unverändert weiterreichen
    - reasoning / text / limits: Abbildung auf Proxy-Parameter (reasoning_effort, verbosity, max_tokens)
    """
    model = str(body.get("model") or "").strip()
    messages: list[dict[str, Any]] = []

    # 1. Instructions als developer/system Message einbinden
    instructions = body.get("instructions")
    if instructions and isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "developer", "content": instructions})

    # 2. Input normalisieren
    raw_input = body.get("input")
    if isinstance(raw_input, str):
        if raw_input.strip():
            messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")

            if itype == "function_call":
                # Modell-Tool-Aufruf aus Vorrunde
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:12]}"
                name = str(item.get("name") or "")
                raw_args = item.get("arguments", "{}")
                if isinstance(raw_args, dict):
                    args_str = json.dumps(raw_args)
                elif isinstance(raw_args, str):
                    args_str = raw_args
                else:
                    args_str = "{}"
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_str,
                            },
                        }
                    ],
                })
            elif itype == "function_call_output":
                # Tool-Ausführungsergebnis
                call_id = str(item.get("call_id") or item.get("id") or "")
                raw_output = item.get("output", "")
                if isinstance(raw_output, (dict, list)):
                    output_str = json.dumps(raw_output, ensure_ascii=False)
                else:
                    output_str = str(raw_output)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output_str,
                })
            elif itype == "message" or "role" in item:
                # Reguläre Nachricht
                role = str(item.get("role") or "user")
                content = extract_text_content(item.get("content"))
                messages.append({"role": role, "content": content})
            else:
                # Unbekannter Typ mit Text-Inhalt
                if "content" in item:
                    content = extract_text_content(item.get("content"))
                    messages.append({"role": "user", "content": content})

    # 3. Tools normalisieren
    raw_tools = body.get("tools") or []
    standardized_tools: list[dict[str, Any]] = []
    if isinstance(raw_tools, list):
        for t in raw_tools:
            if not isinstance(t, dict):
                continue
            ttype = t.get("type", "function")
            if ttype == "function":
                fn = t.get("function")
                if isinstance(fn, dict):
                    standardized_tools.append(t)
                else:
                    # Flaches Format (Name/Description/Parameters auf Tool-Ebene wie bei Codex Responses)
                    standardized_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {}),
                            "strict": t.get("strict", False),
                        },
                    })
            elif ttype == "namespace":
                # Namespace Tools entpacken
                inner_tools = t.get("tools") or t.get("functions") or []
                if isinstance(inner_tools, list):
                    for it in inner_tools:
                        if isinstance(it, dict):
                            fn = it.get("function")
                            if isinstance(fn, dict):
                                standardized_tools.append(it)
                            else:
                                standardized_tools.append({
                                    "type": "function",
                                    "function": {
                                        "name": it.get("name", ""),
                                        "description": it.get("description", ""),
                                        "parameters": it.get("parameters", {}),
                                        "strict": it.get("strict", False),
                                    },
                                })
            elif ttype == "web_search":
                # Nicht emulierter Builtin-Typ
                pass

    tool_choice = body.get("tool_choice")

    # 4. Optionale Parameter & Mapping
    optional: dict[str, Any] = {}
    for field in ["temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"]:
        if field in body:
            optional[field] = body[field]

    # max_output_tokens -> max_tokens
    if "max_output_tokens" in body:
        optional["max_tokens"] = body["max_output_tokens"]
    elif "max_tokens" in body:
        optional["max_tokens"] = body["max_tokens"]

    # reasoning.effort -> reasoning_effort
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if effort and isinstance(effort, str):
            optional["reasoning_effort"] = effort

    # text.verbosity -> verbosity
    text_cfg = body.get("text")
    if isinstance(text_cfg, dict):
        verb = text_cfg.get("verbosity")
        if verb and isinstance(verb, str):
            optional["verbosity"] = verb

    return model, messages, standardized_tools, tool_choice, optional


def _normalize_responses_usage(usage: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Normalisiert das Usage-Dict für die OpenAI Responses API (input_tokens / output_tokens)
    bei voller Abwärtskompatibilität zu prompt_tokens / completion_tokens.
    """
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
        "prompt_tokens": int(input_tokens or 0),
        "completion_tokens": int(output_tokens or 0),
    }


def build_responses_output(
    completion_id: str,
    created_ts: int,
    model: str,
    content: str,
    tool_calls_data: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    """
    Serialisiert das kanonische Completion-Ergebnis in die OpenAI Responses API
    JSON-Antwortstruktur (für non-streaming Requests).
    """
    resp_id = completion_id if completion_id.startswith("resp_") else f"resp_{completion_id}"
    resp_usage = _normalize_responses_usage(usage)
    output_items: list[dict[str, Any]] = []

    if tool_calls_data:
        for tc in tool_calls_data:
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            name = tc.get("name", "")
            raw_args = tc.get("arguments", "{}")
            if isinstance(raw_args, dict):
                args_str = json.dumps(raw_args, ensure_ascii=False)
            elif isinstance(raw_args, str):
                args_str = raw_args
            else:
                args_str = "{}"

            output_items.append({
                "id": f"item_{call_id}",
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": args_str,
            })
    else:
        output_items.append({
            "id": f"msg_{completion_id}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": content,
                }
            ],
        })

    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_ts,
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": resp_usage,
        "error": None,
    }


def build_responses_sse_events(
    completion_id: str,
    created_ts: int,
    model: str,
    content: str,
    tool_calls_data: list[dict[str, Any]],
    usage: dict[str, Any],
    delay_ms: int = 0,
) -> Iterator[str]:
    """
    Erzeugt einen Generator von Server-Sent Events (SSE) gemäß der
    offiziellen OpenAI Responses API Event-Spezifikation für Codex CLI und Desktop App.
    """
    resp_id = completion_id if completion_id.startswith("resp_") else f"resp_{completion_id}"
    resp_usage = _normalize_responses_usage(usage)

    # 1. response.created
    created_event = {
        "type": "response.created",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "in_progress",
            "model": model,
            "output": [],
            "usage": None,
        },
    }
    yield f"event: response.created\ndata: {json.dumps(created_event, ensure_ascii=False)}\n\n"

    # 2. response.in_progress
    in_progress_event = {
        "type": "response.in_progress",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "in_progress",
            "model": model,
            "output": [],
            "usage": None,
        },
    }
    yield f"event: response.in_progress\ndata: {json.dumps(in_progress_event, ensure_ascii=False)}\n\n"

    output_items: list[dict[str, Any]] = []

    if tool_calls_data:
        for idx, tc in enumerate(tool_calls_data):
            call_id = tc.get("id") or f"call_{uuid.uuid4().hex[:12]}"
            name = tc.get("name", "")
            raw_args = tc.get("arguments", "{}")
            if isinstance(raw_args, dict):
                args_str = json.dumps(raw_args, ensure_ascii=False)
            elif isinstance(raw_args, str):
                args_str = raw_args
            else:
                args_str = "{}"
            item_id = f"item_{call_id}"

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            # output_item.added
            ev_added = {
                "type": "response.output_item.added",
                "output_index": idx,
                "item": {
                    "id": item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                },
            }
            yield f"event: response.output_item.added\ndata: {json.dumps(ev_added, ensure_ascii=False)}\n\n"

            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

            # function_call_arguments.delta
            ev_delta = {
                "type": "response.function_call_arguments.delta",
                "output_index": idx,
                "call_id": call_id,
                "delta": args_str,
            }
            yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(ev_delta, ensure_ascii=False)}\n\n"

            # function_call_arguments.done
            ev_fcdone = {
                "type": "response.function_call_arguments.done",
                "output_index": idx,
                "call_id": call_id,
                "arguments": args_str,
            }
            yield f"event: response.function_call_arguments.done\ndata: {json.dumps(ev_fcdone, ensure_ascii=False)}\n\n"

            completed_item = {
                "id": item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call_id,
                "name": name,
                "arguments": args_str,
            }
            ev_itemdone = {
                "type": "response.output_item.done",
                "output_index": idx,
                "item": completed_item,
            }
            yield f"event: response.output_item.done\ndata: {json.dumps(ev_itemdone, ensure_ascii=False)}\n\n"

            output_items.append(completed_item)
    else:
        msg_id = f"msg_{completion_id}"

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # output_item.added
        ev_added = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": msg_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }
        yield f"event: response.output_item.added\ndata: {json.dumps(ev_added, ensure_ascii=False)}\n\n"

        # content_part.added
        ev_cpart = {
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        }
        yield f"event: response.content_part.added\ndata: {json.dumps(ev_cpart, ensure_ascii=False)}\n\n"

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

        # output_text.delta
        ev_delta = {
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": content,
        }
        yield f"event: response.output_text.delta\ndata: {json.dumps(ev_delta, ensure_ascii=False)}\n\n"

        # output_text.done
        ev_tdone = {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "text": content,
        }
        yield f"event: response.output_text.done\ndata: {json.dumps(ev_tdone, ensure_ascii=False)}\n\n"

        # content_part.done
        ev_cpdone = {
            "type": "response.content_part.done",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": content},
        }
        yield f"event: response.content_part.done\ndata: {json.dumps(ev_cpdone, ensure_ascii=False)}\n\n"

        completed_msg = {
            "id": msg_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }
        ev_itemdone = {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": completed_msg,
        }
        yield f"event: response.output_item.done\ndata: {json.dumps(ev_itemdone, ensure_ascii=False)}\n\n"

        output_items.append(completed_msg)

    # response.completed
    ev_completed = {
        "type": "response.completed",
        "response": {
            "id": resp_id,
            "object": "response",
            "created_at": created_ts,
            "status": "completed",
            "model": model,
            "output": output_items,
            "usage": resp_usage,
        },
    }
    yield f"event: response.completed\ndata: {json.dumps(ev_completed, ensure_ascii=False)}\n\n"
