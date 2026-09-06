"""
AcademicAI Configuration Module.

Single Source of Truth (SSOT) for all configuration, environment variables,
defaults, limits, and runtime validation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

load_dotenv()

# Set of known insecure placeholder keys that must not be used
INSECURE_PROXY_KEYS: set[str] = {
    "academicai-proxy",
    "sk-xxx",
    "change-me",
    "changeme",
    "your-secret-key-here",
    "replace-with-strong-key",
}


def _validate_proxy_api_key(api_key: str) -> str:
    """
    Validates that the proxy API key is not empty, not a known insecure placeholder,
    and meets the minimum length requirement (>= 16 characters).
    """
    key = (api_key or "").strip()
    if not key:
        raise RuntimeError("ACADEMICAI_PROXY_API_KEY is required and must not be empty.")
    if key.lower() in {k.lower() for k in INSECURE_PROXY_KEYS}:
        raise RuntimeError("ACADEMICAI_PROXY_API_KEY uses an insecure placeholder value.")
    if len(key) < 16:
        raise RuntimeError("ACADEMICAI_PROXY_API_KEY is too short; use at least 16 characters.")
    return key


def _parse_bool(val: Any, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def _parse_int(val: Any, default: int, min_val: int | None = None) -> int:
    if val is None:
        return default
    try:
        res = int(str(val).strip())
        if min_val is not None:
            res = max(min_val, res)
        return res
    except (ValueError, TypeError):
        return default


def _parse_float(val: Any, default: float, min_val: float | None = None) -> float:
    if val is None:
        return default
    try:
        res = float(str(val).strip())
        if min_val is not None:
            res = max(min_val, res)
        return res
    except (ValueError, TypeError):
        return default


def _parse_list(val: Any, default: list[str]) -> list[str]:
    if val is None:
        return list(default)
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return list(default)
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [item.strip() for item in s.split(",") if item.strip()]


@dataclass
class Settings:
    # Network & Auth
    PORT: int = 11435
    API_KEY: str = "test-proxy-key-123456"
    BASE_URL: str = "https://academic-ai.boku.ac.at/api/v1"
    CLIENT_ID: str = ""
    CLIENT_SECRET: str = ""

    # Health & Cost
    HEALTH_CHECK_BACKEND: bool = True
    HEALTH_CHECK_TIMEOUT_SECONDS: float = 2.0
    ENABLE_COST_MONITORING: bool = True
    COST_CACHE_FILE: str = "data/cost_cache.json"
    COST_CACHE_TTL_SECONDS: int = 600
    COST_REFRESH_TIMEOUT_SECONDS: float = 8.0

    # Payload & Request Limits
    MAX_MESSAGES: int = 200
    MAX_TOOLS: int = 64
    MAX_MESSAGE_TEXT_CHARS: int = 500000
    MAX_TOOL_SCHEMA_CHARS: int = 100000
    MAX_REQUEST_JSON_CHARS: int = 2000000

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 120
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # Generation Defaults
    DEFAULT_CHAT_TEMPERATURE: float = 0.6
    DEFAULT_TOOL_TEMPERATURE: float = 0.1
    DEFAULT_CHAT_VERBOSITY: str = "medium"
    DEFAULT_TOOL_VERBOSITY: str = "low"
    DEFAULT_TOOL_REASONING_EFFORT: str = "low"

    # Humanization
    ENABLE_HUMANIZATION_PASS: bool = False
    HUMANIZATION_MODEL: str = ""
    HUMANIZATION_TEMPERATURE: float = 0.7

    # Streaming & Debug
    STREAM_CHUNK_DELAY_MS: int = 0
    DEBUG_DUMPS: bool = False
    ALLOWED_MODELS: list[str] = field(
        default_factory=lambda: ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini"]
    )

    # Process & Logging
    PID_FILE: Path = field(default_factory=lambda: Path("server.pid"))
    LOG_FILE_PATH: str = "server.log"
    ERR_FILE_PATH: str = "server.err.log"

    # Retries
    RETRY_MAX: int = 2
    RETRY_BASE_MS: int = 300

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = os.environ if env is None else env

        cost_cache_default = str(Path("data") / "cost_cache.json")
        pid_default = Path("server.pid")

        return cls(
            PORT=_parse_int(e.get("ACADEMICAI_PROXY_PORT"), 11435, min_val=1),
            API_KEY=str(e.get("ACADEMICAI_PROXY_API_KEY", "test-proxy-key-123456")).strip(),
            BASE_URL=str(e.get("ACADEMICAI_BASE_URL", "https://academic-ai.boku.ac.at/api/v1")).rstrip("/"),
            CLIENT_ID=str(e.get("ACADEMICAI_CLIENT_ID", "")).strip(),
            CLIENT_SECRET=str(e.get("ACADEMICAI_CLIENT_SECRET", "")).strip(),

            HEALTH_CHECK_BACKEND=_parse_bool(e.get("ACADEMICAI_HEALTH_CHECK_BACKEND"), True),
            HEALTH_CHECK_TIMEOUT_SECONDS=_parse_float(e.get("ACADEMICAI_HEALTH_CHECK_TIMEOUT_SECONDS"), 2.0, min_val=0.2),
            ENABLE_COST_MONITORING=_parse_bool(e.get("ACADEMICAI_ENABLE_COST_MONITORING"), True),
            COST_CACHE_FILE=str(e.get("ACADEMICAI_COST_CACHE_FILE", cost_cache_default)),
            COST_CACHE_TTL_SECONDS=_parse_int(e.get("ACADEMICAI_COST_CACHE_TTL_SECONDS"), 600, min_val=60),
            COST_REFRESH_TIMEOUT_SECONDS=_parse_float(e.get("ACADEMICAI_COST_REFRESH_TIMEOUT_SECONDS"), 8.0, min_val=1.0),

            MAX_MESSAGES=_parse_int(e.get("ACADEMICAI_MAX_MESSAGES"), 200, min_val=1),
            MAX_TOOLS=_parse_int(e.get("ACADEMICAI_MAX_TOOLS"), 64, min_val=0),
            MAX_MESSAGE_TEXT_CHARS=_parse_int(e.get("ACADEMICAI_MAX_MESSAGE_TEXT_CHARS"), 500000, min_val=256),
            MAX_TOOL_SCHEMA_CHARS=_parse_int(e.get("ACADEMICAI_MAX_TOOL_SCHEMA_CHARS"), 100000, min_val=256),
            MAX_REQUEST_JSON_CHARS=_parse_int(e.get("ACADEMICAI_MAX_REQUEST_JSON_CHARS"), 2000000, min_val=1024),

            RATE_LIMIT_PER_MINUTE=_parse_int(e.get("ACADEMICAI_RATE_LIMIT_PER_MINUTE"), 120, min_val=0),
            RATE_LIMIT_WINDOW_SECONDS=_parse_int(e.get("ACADEMICAI_RATE_LIMIT_WINDOW_SECONDS"), 60, min_val=1),

            DEFAULT_CHAT_TEMPERATURE=_parse_float(e.get("ACADEMICAI_DEFAULT_CHAT_TEMPERATURE"), 0.6),
            DEFAULT_TOOL_TEMPERATURE=_parse_float(e.get("ACADEMICAI_DEFAULT_TOOL_TEMPERATURE"), 0.1),
            DEFAULT_CHAT_VERBOSITY=str(e.get("ACADEMICAI_DEFAULT_CHAT_VERBOSITY", "medium")).strip(),
            DEFAULT_TOOL_VERBOSITY=str(e.get("ACADEMICAI_DEFAULT_TOOL_VERBOSITY", "low")).strip(),
            DEFAULT_TOOL_REASONING_EFFORT=str(e.get("ACADEMICAI_DEFAULT_TOOL_REASONING_EFFORT", "low")).strip(),

            ENABLE_HUMANIZATION_PASS=_parse_bool(e.get("ACADEMICAI_ENABLE_HUMANIZATION_PASS"), False),
            HUMANIZATION_MODEL=str(e.get("ACADEMICAI_HUMANIZATION_MODEL", "")).strip(),
            HUMANIZATION_TEMPERATURE=_parse_float(e.get("ACADEMICAI_HUMANIZATION_TEMPERATURE"), 0.7),

            STREAM_CHUNK_DELAY_MS=_parse_int(e.get("ACADEMICAI_STREAM_CHUNK_DELAY_MS"), 0, min_val=0),
            DEBUG_DUMPS=_parse_bool(e.get("ACADEMICAI_DEBUG_DUMPS"), False),
            ALLOWED_MODELS=_parse_list(
                e.get("ACADEMICAI_ALLOWED_MODELS"),
                ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini"],
            ),

            PID_FILE=Path(e.get("ACADEMICAI_PID_FILE", str(pid_default))),
            LOG_FILE_PATH=str(e.get("ACADEMICAI_PROXY_LOG_FILE", "server.log")),
            ERR_FILE_PATH=str(e.get("ACADEMICAI_PROXY_ERR_FILE", "server.err.log")),

            RETRY_MAX=_parse_int(e.get("ACADEMICAI_RETRY_MAX"), 2),
            RETRY_BASE_MS=_parse_int(e.get("ACADEMICAI_RETRY_BASE_MS"), 300),
        )

    def validate(self) -> None:
        """Explicit validation of configuration settings."""
        _validate_proxy_api_key(self.API_KEY)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Returns the cached singleton Settings instance, loaded from environment."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Reloads settings from the environment (or passed dict) and updates singleton & module variables."""
    global _settings, PORT, API_KEY, BASE_URL, CLIENT_ID, CLIENT_SECRET
    global HEALTH_CHECK_BACKEND, HEALTH_CHECK_TIMEOUT_SECONDS, ENABLE_COST_MONITORING
    global COST_CACHE_FILE, COST_CACHE_TTL_SECONDS, COST_REFRESH_TIMEOUT_SECONDS
    global MAX_MESSAGES, MAX_TOOLS, MAX_MESSAGE_TEXT_CHARS, MAX_TOOL_SCHEMA_CHARS, MAX_REQUEST_JSON_CHARS
    global RATE_LIMIT_PER_MINUTE, RATE_LIMIT_WINDOW_SECONDS
    global DEFAULT_CHAT_TEMPERATURE, DEFAULT_TOOL_TEMPERATURE, DEFAULT_CHAT_VERBOSITY
    global DEFAULT_TOOL_VERBOSITY, DEFAULT_TOOL_REASONING_EFFORT
    global ENABLE_HUMANIZATION_PASS, HUMANIZATION_MODEL, HUMANIZATION_TEMPERATURE
    global STREAM_CHUNK_DELAY_MS, DEBUG_DUMPS, ALLOWED_MODELS
    global PID_FILE, LOG_FILE_PATH, ERR_FILE_PATH, RETRY_MAX, MAX_RETRIES, RETRY_BASE_MS

    _settings = Settings.from_env(env)
    for k, v in _settings.__dict__.items():
        globals()[k] = v
    MAX_RETRIES = _settings.RETRY_MAX
    return _settings


def validate_config(settings: Settings | None = None) -> None:
    """Validates configuration explicitly. Raises RuntimeError on validation errors."""
    s = settings or get_settings()
    s.validate()


# Module-level initial exports (lazy / default loaded from environment)
_default_settings = get_settings()

PORT = _default_settings.PORT
API_KEY = _default_settings.API_KEY
BASE_URL = _default_settings.BASE_URL
CLIENT_ID = _default_settings.CLIENT_ID
CLIENT_SECRET = _default_settings.CLIENT_SECRET

HEALTH_CHECK_BACKEND = _default_settings.HEALTH_CHECK_BACKEND
HEALTH_CHECK_TIMEOUT_SECONDS = _default_settings.HEALTH_CHECK_TIMEOUT_SECONDS
ENABLE_COST_MONITORING = _default_settings.ENABLE_COST_MONITORING
COST_CACHE_FILE = _default_settings.COST_CACHE_FILE
COST_CACHE_TTL_SECONDS = _default_settings.COST_CACHE_TTL_SECONDS
COST_REFRESH_TIMEOUT_SECONDS = _default_settings.COST_REFRESH_TIMEOUT_SECONDS

MAX_MESSAGES = _default_settings.MAX_MESSAGES
MAX_TOOLS = _default_settings.MAX_TOOLS
MAX_MESSAGE_TEXT_CHARS = _default_settings.MAX_MESSAGE_TEXT_CHARS
MAX_TOOL_SCHEMA_CHARS = _default_settings.MAX_TOOL_SCHEMA_CHARS
MAX_REQUEST_JSON_CHARS = _default_settings.MAX_REQUEST_JSON_CHARS

RATE_LIMIT_PER_MINUTE = _default_settings.RATE_LIMIT_PER_MINUTE
RATE_LIMIT_WINDOW_SECONDS = _default_settings.RATE_LIMIT_WINDOW_SECONDS

DEFAULT_CHAT_TEMPERATURE = _default_settings.DEFAULT_CHAT_TEMPERATURE
DEFAULT_TOOL_TEMPERATURE = _default_settings.DEFAULT_TOOL_TEMPERATURE
DEFAULT_CHAT_VERBOSITY = _default_settings.DEFAULT_CHAT_VERBOSITY
DEFAULT_TOOL_VERBOSITY = _default_settings.DEFAULT_TOOL_VERBOSITY
DEFAULT_TOOL_REASONING_EFFORT = _default_settings.DEFAULT_TOOL_REASONING_EFFORT

ENABLE_HUMANIZATION_PASS = _default_settings.ENABLE_HUMANIZATION_PASS
HUMANIZATION_MODEL = _default_settings.HUMANIZATION_MODEL
HUMANIZATION_TEMPERATURE = _default_settings.HUMANIZATION_TEMPERATURE

# Deprecated / inert legacy variables (retained as inert fallbacks)
ENABLE_SKILL_SNIPPETS: bool = False
SKILL_SNIPPETS_FILE: str = ""
SKILL_SNIPPETS_MAX: int = 0
ENABLE_AUTO_SKILL_LEARNING: bool = False
AUTO_SKILL_TOPICS_PER_CALL: int = 0
AUTO_SKILL_MIN_TOPIC_LEN: int = 0

STREAM_CHUNK_DELAY_MS = _default_settings.STREAM_CHUNK_DELAY_MS
DEBUG_DUMPS = _default_settings.DEBUG_DUMPS
ALLOWED_MODELS = _default_settings.ALLOWED_MODELS

PID_FILE = _default_settings.PID_FILE
LOG_FILE_PATH = _default_settings.LOG_FILE_PATH
ERR_FILE_PATH = _default_settings.ERR_FILE_PATH

RETRY_MAX = _default_settings.RETRY_MAX
MAX_RETRIES = RETRY_MAX
RETRY_BASE_MS = _default_settings.RETRY_BASE_MS

_INSECURE_PROXY_KEYS = INSECURE_PROXY_KEYS

__all__ = [
    "Settings",
    "get_settings",
    "reload_settings",
    "validate_config",
    "INSECURE_PROXY_KEYS",
    "_INSECURE_PROXY_KEYS",
    "_validate_proxy_api_key",
    "PORT",
    "API_KEY",
    "BASE_URL",
    "CLIENT_ID",
    "CLIENT_SECRET",
    "HEALTH_CHECK_BACKEND",
    "HEALTH_CHECK_TIMEOUT_SECONDS",
    "ENABLE_COST_MONITORING",
    "COST_CACHE_FILE",
    "COST_CACHE_TTL_SECONDS",
    "COST_REFRESH_TIMEOUT_SECONDS",
    "MAX_MESSAGES",
    "MAX_TOOLS",
    "MAX_MESSAGE_TEXT_CHARS",
    "MAX_TOOL_SCHEMA_CHARS",
    "MAX_REQUEST_JSON_CHARS",
    "RATE_LIMIT_PER_MINUTE",
    "RATE_LIMIT_WINDOW_SECONDS",
    "DEFAULT_CHAT_TEMPERATURE",
    "DEFAULT_TOOL_TEMPERATURE",
    "DEFAULT_CHAT_VERBOSITY",
    "DEFAULT_TOOL_VERBOSITY",
    "DEFAULT_TOOL_REASONING_EFFORT",
    "ENABLE_HUMANIZATION_PASS",
    "HUMANIZATION_MODEL",
    "HUMANIZATION_TEMPERATURE",
    "ENABLE_SKILL_SNIPPETS",
    "SKILL_SNIPPETS_FILE",
    "SKILL_SNIPPETS_MAX",
    "ENABLE_AUTO_SKILL_LEARNING",
    "AUTO_SKILL_TOPICS_PER_CALL",
    "AUTO_SKILL_MIN_TOPIC_LEN",
    "STREAM_CHUNK_DELAY_MS",
    "DEBUG_DUMPS",
    "ALLOWED_MODELS",
    "PID_FILE",
    "LOG_FILE_PATH",
    "ERR_FILE_PATH",
    "RETRY_MAX",
    "MAX_RETRIES",
    "RETRY_BASE_MS",
]
