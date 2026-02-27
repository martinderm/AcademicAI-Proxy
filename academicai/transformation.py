"""
Request/Response Transformation: OpenAI (LiteLLM) ↔ AcademicAI
"""

import time
import uuid
from typing import Any


def _normalize_messages(messages: list) -> list:
    """
    Normalisiert Messages für das AcademicAI-Backend:
    - role=system  → in erste user-Message einbauen
    - role=tool    → role=user mit "[Tool result]"-Prefix
    - assistant mit tool_calls → nur Text-Content behalten
    AcademicAI unterstützt nur: user / assistant
    """
    # 1. System-Messages extrahieren
    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    def _extract_text(content) -> str:
        """Extrahiert Text aus string oder list-content (multi-part)."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    # {"type": "text", "text": "..."} oder {"type": "tool_result", "content": "..."}
                    if part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_result":
                        inner = part.get("content", "")
                        parts.append(_extract_text(inner))
                    else:
                        parts.append(str(part.get("text") or part.get("content") or ""))
            return "\n".join(p for p in parts if p)
        return str(content)

    # 2. tool / tool_calls normalisieren
    normalized = []
    for m in non_system:
        role = m.get("role")
        content = _extract_text(m.get("content"))

        if role == "tool":
            # Tool-Result als user-Message (tool_call_id zur Zuordnung mitgeben)
            tool_call_id = m.get("tool_call_id", "")
            id_hint = f" (id: {tool_call_id})" if tool_call_id else ""
            normalized.append({"role": "user", "content": f"[Tool result{id_hint}]\n{content}"})
        elif role == "assistant":
            # tool_calls aus der Gesprächshistorie lesbar darstellen
            tool_calls = m.get("tool_calls") or []
            if tool_calls:
                call_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    tc_name = fn.get("name", "?")
                    tc_args = fn.get("arguments", "{}")
                    call_lines.append(f"[Tool call: {tc_name}({tc_args})]")
                calls_text = "\n".join(call_lines)
                combined = (content + "\n" + calls_text) if content else calls_text
                normalized.append({"role": "assistant", "content": combined})
            elif content:
                normalized.append({"role": "assistant", "content": content})
            # reiner tool_call ohne Text und ohne tool_calls-Feld: überspringen
        else:
            normalized.append({"role": role, "content": content})

    # 3. Aufeinanderfolgende gleiche Rollen zusammenführen
    merged = []
    for m in normalized:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append({"role": m["role"], "content": m["content"]})

    # 4. Muss mit user beginnen
    while merged and merged[0]["role"] != "user":
        merged.pop(0)

    # 5. Auf letzte 20 Turns kürzen (Kontextfenster schonen)
    if len(merged) > 20:
        merged = merged[-20:]
        # Wieder mit user beginnen
        while merged and merged[0]["role"] != "user":
            merged.pop(0)

    # 6. System-Text zuletzt injizieren (sticky), damit er nicht durchs Trimming verloren geht
    if system_parts:
        system_text = "\n\n".join(system_parts)
        injected = False
        result = []
        for m in merged:
            if not injected and m.get("role") == "user":
                merged_content = f"[System context]\n{system_text}\n\n[User message]\n{m['content']}"
                result.append({"role": "user", "content": merged_content})
                injected = True
            else:
                result.append(m)
        if not injected:
            result.insert(0, {"role": "user", "content": system_text})
        merged = result

    return merged


def build_request_body(model: str, messages: list, optional_params: dict) -> dict:
    """
    Wandelt LiteLLM-Parameter in AcademicAI-Request-Body um.

    OpenAI Field              → AcademicAI Field
    ──────────────────────────────────────────────
    model                     → model
    messages                  → messages  (system-messages werden gemergt)
    temperature               → temperature
    max_tokens                → maxTokens
    max_completion_tokens     → maxCompletionTokens
    frequency_penalty         → frequencyPenalty
    presence_penalty          → presencePenalty
    stop                      → stop  (Mistral only, wird trotzdem durchgereicht)
    response_format           → responseFormat
    reasoning_effort          → reasoningEffort
    verbosity                 → verbosity
    tailored_ai_id            → tailoredAiId  (provider-spezifisch / extra_body)
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": _normalize_messages(messages),
    }

    # Einfache 1:1 Felder
    scalar_mapping = {
        "temperature": "temperature",
        "max_tokens": "maxTokens",
        "max_completion_tokens": "maxCompletionTokens",
        "frequency_penalty": "frequencyPenalty",
        "presence_penalty": "presencePenalty",
        "stop": "stop",
        "seed": "seed",
        "reasoning_effort": "reasoningEffort",
        "verbosity": "verbosity",
    }

    for openai_key, academicai_key in scalar_mapping.items():
        if openai_key in optional_params and optional_params[openai_key] is not None:
            body[academicai_key] = optional_params[openai_key]

    # response_format → responseFormat
    if "response_format" in optional_params and optional_params["response_format"] is not None:
        body["responseFormat"] = optional_params["response_format"]

    # tailoredAiId: kann via extra_body übergeben werden
    extra_body = optional_params.get("extra_body") or {}
    if "tailoredAiId" in extra_body:
        body["tailoredAiId"] = extra_body["tailoredAiId"]

    return body


def build_model_response(model: str, academic_response: dict) -> dict:
    """
    Wandelt AcademicAI Chat-Response in OpenAI-kompatibles Format um.

    AcademicAI:
    {
      "data": {
        "content": "...",
        "finishReason": "stop",
        "usage": {
          "promptTokens": 10,
          "completionTokens": 20,
          "totalTokens": 30
        },
        "citations": [...]   // optional, bei RAG
      }
    }
    """
    data = academic_response.get("data", {})
    content = data.get("content", "")
    finish_reason = data.get("finishReason", "stop")
    usage = data.get("usage", {})
    citations = data.get("citations")

    response: dict[str, Any] = {
        "id": f"academicai-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("promptTokens", 0),
            "completion_tokens": usage.get("completionTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
        },
    }

    # Citations als provider_specific durchreichen (kein OpenAI-Standardfeld)
    if citations:
        response["provider_specific"] = {"citations": citations}

    return response


def build_models_response(academic_models: list) -> dict:
    """
    Wandelt AcademicAI /llm/models Response in OpenAI /v1/models Format um.
    """
    models = []
    for m in academic_models:
        model_entry: dict[str, Any] = {
            "id": m.get("modelName", ""),
            "object": "model",
            "created": 0,
            "owned_by": "academicai",
        }
        # Optionale Metadaten
        if "contextWindow" in m:
            model_entry["context_window"] = m["contextWindow"]
        if "outputTokenLimit" in m:
            model_entry["max_tokens"] = m["outputTokenLimit"]
        if "costs" in m:
            model_entry["costs"] = m["costs"]
        models.append(model_entry)

    return {
        "object": "list",
        "data": models,
    }
