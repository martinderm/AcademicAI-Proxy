"""
Security helpers for redacting sensitive data in logs and debug dumps.
"""

from __future__ import annotations

import re
from typing import Any

_SENSITIVE_KEY_PARTS = (
    "authorization",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "client_secret",
    "x-client-secret",
)

_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._\-~=+/]+")


def _looks_sensitive_key(name: str) -> bool:
    key = str(name or "").lower()
    return any(part in key for part in _SENSITIVE_KEY_PARTS)


def redact_sensitive(value: Any) -> Any:
    """Recursively redacts likely secrets from dict/list/string values."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if _looks_sensitive_key(str(k)):
                out[k] = "***REDACTED***"
            else:
                out[k] = redact_sensitive(v)
        return out

    if isinstance(value, list):
        return [redact_sensitive(v) for v in value]

    if isinstance(value, str):
        return _BEARER_RE.sub("Bearer ***REDACTED***", value)

    return value
