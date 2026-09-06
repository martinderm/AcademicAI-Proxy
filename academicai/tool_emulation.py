"""
Tool-Call-Emulation fuer AcademicAI (Strategie A: JSON-Mode)

AcademicAI unterstuetzt kein natives Function-Calling.
Dieses Modul emuliert den vollen OpenAI-Tool-Call-Flow via JSON-Mode:

  1. Tool-Definitionen + JSON-Entscheidungsschema als System-Prompt injizieren
  2. response_format: {type: "json_object"} an Backend senden
  3. Modell antwortet mit:
       {"action": "tool_call", "name": "...", "arguments": {...}}
     oder:
       {"action": "respond", "content": "..."}
  4. Proxy parst JSON -> entweder OpenAI tool_calls Response oder normaler Text
  5. Streaming: Tool-Call-Chunks im OpenAI-SSE-Format ausgeben

OpenClaw uebernimmt die taatsaechliche Tool-Ausfuehrung und sendet
das Ergebnis als role=tool zurueck -- der Proxy leistet nur Formatkonvertierung.
"""

import json
import re
import uuid
from typing import Optional, Any


def _repair_json_str(raw: str) -> str:
    """Bereinigt gängige LLM-JSON-Syntaxfehler wie Trailing Commas vor } oder ]."""
    return re.sub(r",+\s*([\}\]])", r"\1", raw)


def _repair_and_load_json(raw: str) -> Optional[Any]:
    """
    Parst JSON robust mit mehrstufiger Reparatur:
    1. Standard json.loads
    2. json.loads mit strict=False (erlaubt unescapte Steuerzeichen/Newlines/Tabs in Strings)
    3. Trailing-Comma-Bereinigung vor } oder ] (,\\s*([}\\]]) -> \\1)
    4. Reparatur ungültiger Backslash-Escapes (z.B. Windows-Pfade C:\\users\\...)
    """
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None

    # 1. Direkter Parse (Normalfall, schnell)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. strict=False (erlaubt unescapte Steuerzeichen/Newlines in Strings)
    try:
        return json.loads(s, strict=False)
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Trailing Commas bereinigen
    repaired = _repair_json_str(s)
    try:
        return json.loads(repaired)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return json.loads(repaired, strict=False)
    except (json.JSONDecodeError, ValueError):
        pass

    # 4. Ungültige Escape-Sequenzen reparieren
    try:
        fixed_escapes = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', repaired)
        fixed_escapes = re.sub(r'\\([^"\\/bfnrtu])', r'\\\\\1', fixed_escapes)
        return json.loads(fixed_escapes, strict=False)
    except (json.JSONDecodeError, ValueError):
        pass

    return None


def _parse_json_loose(content: str):
    """Parst JSON robust: raw JSON oder ```json ...``` Codefence mit Reparatur."""
    if not content or not isinstance(content, str):
        return None
    s = content.strip()

    parsed = _repair_and_load_json(s)
    if parsed is not None:
        return parsed

    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", s, re.IGNORECASE | re.DOTALL)
    if fence:
        inner = fence.group(1).strip()
        parsed = _repair_and_load_json(inner)
        if parsed is not None:
            return parsed

    return None


# ---------------------------------------------------------------------------
# System-Prompt fuer JSON-Mode
# ---------------------------------------------------------------------------

_TOOL_SYSTEM_TEMPLATE = """\
You have access to the following tools. When the user's request requires \
external information, actions, or data you cannot know from training, \
you MUST use the appropriate tool.

Available tools:
{tool_list}

You MUST respond with a valid JSON object. No explanation, no markdown, \
no extra text outside the JSON. Choose one of these formats:

If one tool is needed:
{{"action": "tool_call", "name": "<tool_name>", "arguments": {{<params>}}}}

If multiple tool calls are needed in one turn:
{{"action": "tool_calls", "calls": [
  {{"name": "<tool_name>", "arguments": {{<params>}}}},
  {{"name": "<tool_name>", "arguments": {{<params>}}}}
]}}

If you can answer directly from your knowledge:
{{"action": "respond", "content": "<your answer>"}}
"""


def _format_default_val(val: Any) -> str:
    """Formatiert einen Default-Wert für TypeScript-Signaturen."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return json.dumps(val)
    try:
        return json.dumps(val)
    except Exception:
        return str(val)


def _format_param_type(pdef: dict) -> str:
    """Formatiert den Typ eines Parameters im TypeScript-Stil."""
    enum_vals = pdef.get("enum")
    if enum_vals and isinstance(enum_vals, (list, tuple)):
        if len(enum_vals) > 4:
            items = [json.dumps(x) for x in enum_vals[:4]] + ["..."]
        else:
            items = [json.dumps(x) for x in enum_vals]
        return " | ".join(items)

    raw_type = pdef.get("type", "any")
    if raw_type == "integer":
        return "number"
    elif raw_type == "array":
        items = pdef.get("items")
        if isinstance(items, dict):
            item_type = items.get("type")
            if item_type == "integer":
                item_type = "number"
            if item_type:
                return f"{item_type}[]"
        return "any[]"
    elif raw_type == "object":
        props = pdef.get("properties")
        if isinstance(props, dict) and props:
            keys = ", ".join(props.keys())
            return f"{{{keys}}}"
        return "object"
    return str(raw_type)


def _compact_tool_def(tool: dict) -> str:
    """Einzeilige TypeScript-aehnliche Signatur eines Tools fuer den System-Prompt."""
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return ""
    fn = tool.get("function")
    if not isinstance(fn, dict):
        return ""
    name = fn.get("name", "?")
    desc = (fn.get("description") or "").split("\n")[0][:120].strip()

    params_obj = fn.get("parameters") or {}
    params = params_obj.get("properties") or {}
    required = set(params_obj.get("required") or [])

    param_parts = []
    for pname, pdef in params.items():
        if not isinstance(pdef, dict):
            pdef = {}
        opt = "" if pname in required else "?"
        ptype = _format_param_type(pdef)
        default_str = ""
        if "default" in pdef and pdef["default"] is not None:
            default_str = f" = {_format_default_val(pdef['default'])}"
        param_parts.append(f"{pname}{opt}: {ptype}{default_str}")

    param_str = ", ".join(param_parts)
    desc_str = f" -- {desc}" if desc else ""
    return f"- {name}({param_str}){desc_str}"


def _extract_specific_tool_name(tool_choice: Any) -> Optional[str]:
    """Extrahiert den Tool-Namen, falls tool_choice ein bestimmtes Tool vorgibt."""
    if not tool_choice:
        return None
    if isinstance(tool_choice, str):
        choice = tool_choice.strip()
        if choice.lower() not in ("auto", "none", "required"):
            return choice
    elif isinstance(tool_choice, dict):
        fn = tool_choice.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            return str(fn["name"]).strip()
        if tool_choice.get("name"):
            return str(tool_choice["name"]).strip()
    return None


def build_json_mode_system_prompt(tools: list, tool_choice: Any = None) -> str:
    """Baut den vollstaendigen System-Prompt fuer JSON-Mode Tool-Emulation mit tool_choice-Unterstuetzung."""
    tool_lines = [_compact_tool_def(t) for t in tools if _compact_tool_def(t)]
    tool_list = "\n".join(tool_lines) if tool_lines else "(none)"
    prompt = _TOOL_SYSTEM_TEMPLATE.format(tool_list=tool_list)

    specific_tool = _extract_specific_tool_name(tool_choice)
    if tool_choice == "required":
        prompt += (
            "\n\nCRITICAL INSTRUCTION - MANDATORY TOOL CALL:\n"
            "A tool call is MANDATORY for this request. You are FORBIDDEN from responding with "
            '{"action": "respond"}. You MUST choose and call an appropriate tool.'
        )
    elif specific_tool:
        prompt += (
            "\n\nCRITICAL INSTRUCTION - SPECIFIC TOOL MANDATORY:\n"
            f"You MUST call the specific tool '{specific_tool}'. You are FORBIDDEN from responding with "
            f'{{"action": "respond"}} or calling any other tool. You MUST execute a tool call for "{specific_tool}".'
        )
    elif tool_choice == "none":
        prompt += (
            "\n\nCRITICAL INSTRUCTION - NO TOOL CALLS:\n"
            "You MUST NOT call any tools. You MUST respond with "
            '{"action": "respond", "content": "<your answer>"}.'
        )

    return prompt


def inject_tools_into_messages(messages: list, tools: list, tool_choice: Any = None) -> list:
    """
    Fuegt Tool-Definitionen als System-Message an den Anfang ein.
    transformation.py mergt sie automatisch in die erste User-Message.

    Ausserdem wird eine JSON-Erinnerung ans Ende gehaengt, damit das letzte
    User-Message das Wort 'json' enthaelt -- AcademicAI-Constraint bei
    responseFormat: json_object.
    """
    if not tools:
        return messages
    prompt = build_json_mode_system_prompt(tools, tool_choice=tool_choice)

    specific_tool = _extract_specific_tool_name(tool_choice)
    if tool_choice == "required":
        reminder_content = 'Respond with a JSON tool call as specified above. A tool call is mandatory; {"action": "respond"} is forbidden.'
    elif specific_tool:
        reminder_content = f'Respond with a JSON tool call calling tool "{specific_tool}". You MUST call this specific tool; {{"action": "respond"}} is forbidden.'
    elif tool_choice == "none":
        reminder_content = 'Respond with JSON {"action": "respond", "content": "..."}. Do not call any tools.'
    else:
        reminder_content = "Respond with the JSON format as specified above."

    json_reminder = {"role": "user", "content": reminder_content}
    return [{"role": "system", "content": prompt}] + messages + [json_reminder]


# ---------------------------------------------------------------------------
# Response-Parsing
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[dict]:
    """
    Robuste JSON-Extraktion aus einem Text. Versucht in dieser Reihenfolge:
    1. Direktes _repair_and_load_json (Normalfall)
    2. Markdown-Codeblock: ```json ... ``` oder ``` ... ```
    3. Erstes vollstaendiges {...}-Objekt im Fliesstext per Klammer-Zaehler
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    # 1. Direkter Parse (Normalfall)
    data = _repair_and_load_json(text)
    if isinstance(data, dict):
        return data

    # 2. Markdown-Codeblock: ```json { ... } ``` oder ``` { ... } ```
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        block = match.group(1).strip()
        data = _repair_and_load_json(block)
        if isinstance(data, dict):
            return data
        start = block.find("{")
        end = block.rfind("}")
        if start >= 0 and end > start:
            data = _repair_and_load_json(block[start : end + 1])
            if isinstance(data, dict):
                return data

    # 3. Erstes vollstaendiges {...}-Objekt per Klammer-Zaehler finden
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape_next = False
        for i, c in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if c == "\\" and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1]
                        data = _repair_and_load_json(candidate)
                        if isinstance(data, dict):
                            return data
                        break

    return None


def _normalize_tool_call_entry(entry: dict) -> Optional[dict]:
    """Normalisiert einen einzelnen Tool-Call-Eintrag auf {name, arguments}."""
    if not isinstance(entry, dict):
        return None
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    arguments = entry.get("arguments", {})
    if isinstance(arguments, str):
        parsed = _repair_and_load_json(arguments)
        if isinstance(parsed, dict):
            arguments = parsed
        else:
            arguments = {}
    elif not isinstance(arguments, dict):
        arguments = {}
    return {"name": name.strip(), "arguments": arguments}


def parse_json_mode_response(content: str) -> Optional[dict]:
    """
    Parst die JSON-Mode-Antwort des Modells.

    Returns eines von:
      {"action": "tool_call", "name": str, "arguments": dict}
      {"action": "tool_calls", "calls": list[dict{name,arguments}]}
      {"action": "respond",   "content": str}
      None  -- Parsing fehlgeschlagen (Fallback: Content als Text behandeln)
    """
    if not content:
        return None
    data = _extract_json_object(content)
    if data is None:
        return None

    action = data.get("action")

    if action == "tool_call":
        normalized = _normalize_tool_call_entry(data)
        if normalized is None:
            return None
        return {"action": "tool_call", "name": normalized["name"], "arguments": normalized["arguments"]}

    if action == "tool_calls":
        raw_calls = data.get("calls", [])
        if not isinstance(raw_calls, list):
            return None
        calls = []
        for c in raw_calls:
            normalized = _normalize_tool_call_entry(c)
            if normalized is not None:
                calls.append(normalized)
        if not calls:
            return None
        return {"action": "tool_calls", "calls": calls}

    if action == "respond":
        content_val = data.get("content", "")
        return {"action": "respond", "content": str(content_val)}

    return None


def parse_tool_calls(content: str) -> list[dict]:
    """
    Liefert eine Liste normalisierter Tool-Calls.

    Unterstützt beide JSON-Mode-Formate:
    - action=tool_call (single)
    - action=tool_calls (multi)
    """
    result = parse_json_mode_response(content)
    if not result:
        return []
    if result.get("action") == "tool_call":
        return [{"name": result["name"], "arguments": result.get("arguments", {})}]
    if result.get("action") == "tool_calls":
        return list(result.get("calls") or [])
    return []


def parse_tool_call(content: str) -> Optional[dict]:
    """Legacy-Helper: gibt den ersten Tool-Call zurück (falls vorhanden)."""
    calls = parse_tool_calls(content)
    return calls[0] if calls else None


def extract_respond_content(content: str) -> Optional[str]:
    """
    Extrahiert den Text-Content aus einer JSON-Mode-Antwort.

    Primär: {"action": "respond", "content": "..."}
    Fallback: Beliebiger JSON-Dict ohne action=tool_call —
              sucht nach gängigen Text-Keys (content, answer, text, message, response).
              Damit werden Modelle abgedeckt, die das Schema leicht abwandeln.
    Returns None nur wenn kein gültiges JSON oder action=tool_call erkannt.
    """
    result = parse_json_mode_response(content)
    if result is not None:
        if result.get("action") == "respond":
            return result.get("content")
        # tool_call bleibt tool_call — kein Fallback
        if result.get("action") == "tool_call":
            return None

    # Fallback: valides JSON, aber kein bekanntes Schema
    if not content:
        return None
    data = _extract_json_object(content)
    if data is None:
        return None

    # action=tool_call explizit ausschliessen (Sicherheit)
    if data.get("action") == "tool_call":
        return None

    # Gängige Text-Keys in Prioritätsreihenfolge
    for key in ("content", "answer", "text", "message", "response", "reply", "output"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return None


def format_arbitrary_json_as_codeblock(content: str) -> Optional[str]:
    """
    Letzter Fallback fuer den Fall dass das Modell valides JSON ausgibt,
    das weder action=tool_call noch action=respond enthaelt
    (Modell hat das Schema nicht befolgt, z.B. gibt einen JSON-Plan aus).

    Gibt das JSON als Markdown-Codeblock zurueck.
    Returns None wenn der Content kein gueltiges JSON ist.
    """
    if not content:
        return None
    parsed = _parse_json_loose(content)
    if parsed is None:
        return None
    if isinstance(parsed, dict) and parsed.get("action") in ("tool_call", "respond"):
        return None
    return "```json\n" + json.dumps(parsed, ensure_ascii=False, indent=2) + "\n```"


def format_arbitrary_json_for_humans(content: str) -> Optional[str]:
    """
    Formatiert beliebiges JSON in menschenlesbaren Text (ohne JSON-Codeblock).
    Nur fuer Human-Channels gedacht.

    Ziel: Keine technischen Meta-Felder (status/source/timestamp/...) in der
    sichtbaren Antwort, sondern vorrangig das inhaltliche Ergebnis.
    """
    if not content:
        return None
    parsed = _parse_json_loose(content)
    if parsed is None:
        return None

    if isinstance(parsed, dict) and parsed.get("action") in ("tool_call", "respond"):
        return None

    META_KEYS = {
        "status", "source", "timestamp", "stored_in", "requested_room",
        "note", "path", "id", "ids", "debug", "meta", "metadata",
    }

    def _is_scalar(v):
        return isinstance(v, (str, int, float, bool)) or v is None

    def _fmt_scalar(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "ja" if v else "nein"
        return str(v)

    def _filter_obj(obj):
        if isinstance(obj, dict):
            filtered = {k: _filter_obj(v) for k, v in obj.items() if k not in META_KEYS}
            # Falls alles rausgefiltert wurde: Original behalten (sonst leere Antwort)
            return filtered if filtered else obj
        if isinstance(obj, list):
            return [_filter_obj(x) for x in obj]
        return obj

    parsed = _filter_obj(parsed)

    # Spezialfall: typisches rooms/zoom_rooms-Objekt -> direkte menschenlesbare Liste
    if isinstance(parsed, dict):
        room_list = None
        for key in ("zoom_rooms", "rooms"):
            if isinstance(parsed.get(key), list):
                room_list = parsed[key]
                break
        if room_list is not None:
            lines = ["Hier die bekannten Räume:"]
            for r in room_list:
                if isinstance(r, dict):
                    name = r.get("name")
                    url = r.get("url")
                    if name and url:
                        lines.append(f"- {name}: {url}")
                    elif name:
                        lines.append(f"- {name}")
            if len(lines) > 1:
                return "\n".join(lines)

    def _render(obj, indent=0):
        pad = "  " * indent
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _is_scalar(v):
                    lines.append(f"{pad}- {k}: {_fmt_scalar(v)}")
                elif isinstance(v, list):
                    if not v:
                        continue
                    lines.append(f"{pad}- {k}:")
                    if all(_is_scalar(x) for x in v):
                        for x in v:
                            lines.append(f"{pad}  - {_fmt_scalar(x)}")
                    else:
                        for i, x in enumerate(v, 1):
                            lines.append(f"{pad}  - Eintrag {i}:")
                            lines.extend(_render(x, indent + 2))
                else:
                    lines.append(f"{pad}- {k}:")
                    lines.extend(_render(v, indent + 1))
        elif isinstance(obj, list):
            if not obj:
                return lines
            if all(_is_scalar(x) for x in obj):
                for x in obj:
                    lines.append(f"{pad}- {_fmt_scalar(x)}")
            else:
                for i, x in enumerate(obj, 1):
                    lines.append(f"{pad}- Eintrag {i}:")
                    lines.extend(_render(x, indent + 1))
        else:
            lines.append(f"{pad}- {_fmt_scalar(obj)}")
        return lines

    rendered = _render(parsed)
    if not rendered:
        return None
    return "Hier die Infos:\n" + "\n".join(rendered)


def strip_tool_call_tag(content: str) -> str:
    """Kompatibilitaets-Stub -- bei JSON-Mode nicht benoetigt."""
    return content.strip()


# ---------------------------------------------------------------------------
# Response-Builder (unveraendert)
# ---------------------------------------------------------------------------

def build_tool_calls_response(
    completion_id: str,
    created_ts: int,
    model: str,
    tool_call_data,
    usage: dict,
) -> dict:
    """
    Baut eine vollstaendige OpenAI chat.completion Response mit tool_calls.
    Unterstützt einen einzelnen Dict-Call oder eine Liste von Calls.
    finish_reason = "tool_calls"
    """
    calls_in = tool_call_data if isinstance(tool_call_data, list) else [tool_call_data]
    tool_calls = []
    for c in calls_in:
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        arguments = (c or {}).get("arguments", {})
        arguments_str = (
            json.dumps(arguments, ensure_ascii=False)
            if isinstance(arguments, dict)
            else str(arguments)
        )
        tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": arguments_str,
                },
            }
        )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# SSE-Chunks (Streaming, unveraendert)
# ---------------------------------------------------------------------------

def build_tool_calls_sse_chunks(
    completion_id: str,
    created_ts: int,
    model: str,
    tool_call_data,
) -> list:
    """
    Baut die SSE-Chunk-Sequenz fuer Tool-Calls im OpenAI-Streaming-Format.
    Unterstützt einen einzelnen Dict-Call oder eine Liste von Calls.
    """
    calls_in = tool_call_data if isinstance(tool_call_data, list) else [tool_call_data]
    tool_calls_delta = []
    for idx, c in enumerate(calls_in):
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        arguments = (c or {}).get("arguments", {})
        arguments_str = (
            json.dumps(arguments, ensure_ascii=False)
            if isinstance(arguments, dict)
            else str(arguments)
        )
        tool_calls_delta.append(
            {
                "index": idx,
                "id": call_id,
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": arguments_str,
                },
            }
        )

    return [
        # Chunk 1: role
        {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created_ts, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}],
        },
        # Chunk 2: tool_call delta(s)
        {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created_ts, "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": tool_calls_delta},
                    "finish_reason": None,
                }
            ],
        },
        # Chunk 3: finish
        {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created_ts, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    ]
