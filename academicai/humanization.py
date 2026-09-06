"""
AcademicAI Humanization Module.

Kapselt:
- Erkennung menschlicher Zielkanäle (WhatsApp, Telegram, Discord, OpenClaw etc.)
- Extraktion des letzten User-Prompts aus Message-Historien
- Prompt-Konstruktion für den optionalen zweiten Humanisierungs-Pass
- Asynchrone Ausführung des zweiten Passes (run_humanization_pass) via run_in_threadpool
- Dynamische Konfigurationsauflösung über server / academicai.config
"""

from __future__ import annotations

import inspect
import sys
from typing import Any, Optional

from starlette.concurrency import run_in_threadpool

from academicai.logging_config import log
from academicai.request_guards import extract_text_content


def last_user_text(messages: list) -> str:
    """Liefert den letzten User-Text aus den Original-Messages."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return extract_text_content(m.get("content"))
    return ""


_last_user_text = last_user_text


def is_human_readable_target(messages: list) -> bool:
    """
    Heuristik: Nur bei menschlichen Zielkanälen JSON->Human-Text-Fallback aktivieren.

    False für klar maschinelle Runs (z.B. cron).
    True für typische Human-Channels (whatsapp/telegram/signal/discord/slack/webchat...).
    """
    user_text = "\n".join(
        extract_text_content(m.get("content"))
        for m in (messages or [])
        if isinstance(m, dict) and m.get("role") == "user"
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
        extract_text_content(m.get("content"))
        for m in (messages or [])
        if isinstance(m, dict) and m.get("role") == "system"
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


_is_human_readable_target = is_human_readable_target


def build_humanization_messages(original_user_query: str, structured_content: str) -> list:
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
            f"Original user question:\n{(original_user_query or '').strip() or '-'}\n\n"
            f"Structured/tool-derived result:\n{(structured_content or '').strip()}\n\n"
            "Task: Write a concise, natural-language final reply for the user."
        ),
    }
    return [system_msg, user_msg]


_build_humanization_messages = build_humanization_messages


def _resolve_humanization_model(default_model: str) -> str:
    server_mod = sys.modules.get("server")
    if server_mod is not None and hasattr(server_mod, "HUMANIZATION_MODEL"):
        val = getattr(server_mod, "HUMANIZATION_MODEL")
        if val:
            return val
    from academicai import config
    val = getattr(config, "HUMANIZATION_MODEL", "")
    return val or default_model


def _resolve_humanization_temperature() -> float:
    server_mod = sys.modules.get("server")
    if server_mod is not None and hasattr(server_mod, "HUMANIZATION_TEMPERATURE"):
        val = getattr(server_mod, "HUMANIZATION_TEMPERATURE")
        if val is not None:
            return val
    from academicai import config
    val = getattr(config, "HUMANIZATION_TEMPERATURE", None)
    if val is not None:
        return val
    return 0.7


def _resolve_completion_func() -> Any:
    server_mod = sys.modules.get("server")
    if server_mod is not None:
        server_academicai = getattr(server_mod, "academicai", None)
        if server_academicai is not None and hasattr(server_academicai, "completion"):
            return getattr(server_academicai, "completion")
        if hasattr(server_mod, "completion"):
            return getattr(server_mod, "completion")
    import academicai
    return getattr(academicai, "completion", None)


async def run_humanization_pass(
    model: str, original_user_query: str, structured_content: str
) -> Optional[str]:
    """Führt optionalen zweiten LLM-Pass aus und liefert finalen Text zurück."""
    try:
        human_model = _resolve_humanization_model(model)
        temp = _resolve_humanization_temperature()
        comp_func = _resolve_completion_func()
        if comp_func is None:
            log.warning("humanization pass failed: completion function not found")
            return None

        msgs = build_humanization_messages(original_user_query, structured_content)
        if inspect.iscoroutinefunction(comp_func):
            resp = await comp_func(
                model=human_model,
                messages=msgs,
                temperature=temp,
            )
        else:
            resp = await run_in_threadpool(
                comp_func,
                model=human_model,
                messages=msgs,
                temperature=temp,
            )

        text = ""
        if resp is not None:
            if hasattr(resp, "choices") and resp.choices:
                first_choice = resp.choices[0]
                msg = getattr(first_choice, "message", None)
                if msg is not None:
                    text = getattr(msg, "content", "") or ""
                elif isinstance(first_choice, dict):
                    text = first_choice.get("message", {}).get("content", "") or ""
            elif isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                first_choice = resp["choices"][0]
                if isinstance(first_choice, dict):
                    msg = first_choice.get("message", {})
                    if isinstance(msg, dict):
                        text = msg.get("content", "") or ""
                    else:
                        text = getattr(msg, "content", "") or ""
                else:
                    text = getattr(first_choice, "message", {}).get("content", "") or ""

        text = (text or "").strip()
        return text or None
    except Exception as e:
        log.warning(f"humanization pass failed, fallback to first-pass content: {e}")
        return None


_run_humanization_pass = run_humanization_pass


__all__ = [
    "build_humanization_messages",
    "_build_humanization_messages",
    "is_human_readable_target",
    "_is_human_readable_target",
    "last_user_text",
    "_last_user_text",
    "run_humanization_pass",
    "_run_humanization_pass",
]
