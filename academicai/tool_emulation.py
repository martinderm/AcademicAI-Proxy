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
from typing import Optional


def _parse_json_loose(content: str):
    """Parst JSON robust: raw JSON oder ```json ...``` Codefence."""
    if not content:
        return None
    s = content.strip()

    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```$", s, re.IGNORECASE | re.DOTALL)
    if fence:
        inner = fence.group(1).strip()
        try:
            return json.loads(inner)
        except (json.JSONDecodeError, ValueError):
            return None

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
no extra text outside the JSON. Choose exactly one of these formats:

If a tool is needed:
{{"action": "tool_call", "name": "<tool_name>", "arguments": {{<params>}}}}

If you can answer directly from your knowledge:
{{"action": "respond", "content": "<your answer>"}}

IMPORTANT — exec tool on Windows/PowerShell:
- Commands are already executed inside PowerShell. Do NOT wrap them in \
`powershell -Command "..."` with double quotes (causes $variable expansion).
- Write PowerShell commands directly, e.g.: `Get-Content file.txt`
- If you must use $variables inside a nested call, use single-quoted strings \
or backtick-escape: `$var` → `` `$var ``.
"""


def _compact_tool_def(tool: dict) -> str:
    """Einzeilige Darstellung eines Tools fuer den System-Prompt."""
    if tool.get("type") != "function":
        return ""
    fn = tool["function"]
    name = fn.get("name", "?")
    desc = fn.get("description", "").split("\n")[0][:120]
    params = fn.get("parameters", {}).get("properties", {})
    required = set(fn.get("parameters", {}).get("required", []))
    param_parts = []
    for pname, pdef in params.items():
        ptype = pdef.get("type", "any")
        opt = "" if pname in required else "?"
        param_parts.append(f"{pname}{opt}: {ptype}")
    param_str = ", ".join(param_parts) if param_parts else ""
    return f"- {name}({param_str}) -- {desc}"


def build_json_mode_system_prompt(tools: list) -> str:
    """Baut den vollstaendigen System-Prompt fuer JSON-Mode Tool-Emulation."""
    tool_lines = [_compact_tool_def(t) for t in tools if _compact_tool_def(t)]
    tool_list = "\n".join(tool_lines) if tool_lines else "(none)"
    return _TOOL_SYSTEM_TEMPLATE.format(tool_list=tool_list)


def inject_tools_into_messages(messages: list, tools: list) -> list:
    """
    Fuegt Tool-Definitionen als System-Message an den Anfang ein.
    transformation.py mergt sie automatisch in die erste User-Message.

    Ausserdem wird eine JSON-Erinnerung ans Ende gehaengt, damit das letzte
    User-Message das Wort 'json' enthaelt -- AcademicAI-Constraint bei
    responseFormat: json_object.
    """
    if not tools:
        return messages
    prompt = build_json_mode_system_prompt(tools)
    # Erinnerung ans Ende: wird via _normalize_messages in die letzte
    # User-Message gemergt (consecutive same-role merge in Step 4)
    json_reminder = {"role": "user", "content": "Respond with the JSON format as specified above."}
    return [{"role": "system", "content": prompt}] + messages + [json_reminder]


# ---------------------------------------------------------------------------
# Response-Parsing
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[dict]:
    """
    Robuste JSON-Extraktion aus einem Text. Versucht in dieser Reihenfolge:
    1. Direktes json.loads (Normalfall)
    2. JSON aus Markdown-Codeblock (```json ... ``` oder ``` ... ```)
    3. Erstes vollstaendiges {...}-Objekt im Fliesstext

    Damit werden Modelle abgedeckt, die trotz JSON-Mode-Anweisung
    Erklaerungstext vor/nach dem JSON ausgeben.
    """
    text = text.strip()
    if not text:
        return None

    # 1. Direkter Parse (Normalfall, kein Overhead)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Markdown-Codeblock: ```json { ... } ``` oder ``` { ... } ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass

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
                        try:
                            data = json.loads(text[start : i + 1])
                            if isinstance(data, dict):
                                return data
                        except (json.JSONDecodeError, ValueError):
                            pass
                        break

    return None


def parse_json_mode_response(content: str) -> Optional[dict]:
    """
    Parst die JSON-Mode-Antwort des Modells.

    Returns eines von:
      {"action": "tool_call", "name": str, "arguments": dict}
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
        name = data.get("name")
        if not name or not isinstance(name, str):
            return None
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            # Manchmal liefert das Modell arguments als JSON-String
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {}
        return {"action": "tool_call", "name": name, "arguments": arguments}

    if action == "respond":
        content_val = data.get("content", "")
        return {"action": "respond", "content": str(content_val)}

    return None


def parse_tool_call(content: str) -> Optional[dict]:
    """
    Kompatibilitaets-Wrapper fuer server.py.
    Gibt {"name": str, "arguments": dict} zurueck wenn tool_call,
    sonst None.
    """
    result = parse_json_mode_response(content)
    if result and result.get("action") == "tool_call":
        return {"name": result["name"], "arguments": result.get("arguments", {})}
    return None


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
    """
    if not content:
        return None
    parsed = _parse_json_loose(content)
    if parsed is None:
        return None

    if isinstance(parsed, dict) and parsed.get("action") in ("tool_call", "respond"):
        return None

    def _is_scalar(v):
        return isinstance(v, (str, int, float, bool)) or v is None

    def _fmt_scalar(v):
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "ja" if v else "nein"
        return str(v)

    def _render(obj, indent=0):
        pad = "  " * indent
        lines = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                if _is_scalar(v):
                    lines.append(f"{pad}- {k}: {_fmt_scalar(v)}")
                elif isinstance(v, list):
                    lines.append(f"{pad}- {k}:")
                    if not v:
                        lines.append(f"{pad}  - (leer)")
                    elif all(_is_scalar(x) for x in v):
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
                lines.append(f"{pad}- (leer)")
            elif all(_is_scalar(x) for x in obj):
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
    return "Hier die Infos kompakt:\n" + "\n".join(rendered)


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
    tool_call_data: dict,
    usage: dict,
) -> dict:
    """
    Baut eine vollstaendige OpenAI chat.completion Response mit tool_calls.
    finish_reason = "tool_calls"
    """
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    arguments = tool_call_data.get("arguments", {})
    arguments_str = (
        json.dumps(arguments, ensure_ascii=False)
        if isinstance(arguments, dict)
        else str(arguments)
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
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tool_call_data["name"],
                                "arguments": arguments_str,
                            },
                        }
                    ],
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
    tool_call_data: dict,
) -> list:
    """
    Baut die SSE-Chunk-Sequenz fuer einen Tool-Call im OpenAI-Streaming-Format.
    """
    call_id = f"call_{uuid.uuid4().hex[:24]}"
    arguments = tool_call_data.get("arguments", {})
    arguments_str = (
        json.dumps(arguments, ensure_ascii=False)
        if isinstance(arguments, dict)
        else str(arguments)
    )
    return [
        # Chunk 1: role
        {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created_ts, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}],
        },
        # Chunk 2: tool_call delta
        {
            "id": completion_id, "object": "chat.completion.chunk",
            "created": created_ts, "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": tool_call_data["name"],
                                    "arguments": arguments_str,
                                },
                            }
                        ]
                    },
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
