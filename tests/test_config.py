"""
Unit tests for academicai.config module and server backward compatibility.
Validates:
- Default values when no environment variables are set
- Custom environment variables override defaults
- Safe import: importing academicai.config or server without env vars does not crash
- _validate_proxy_api_key rejects insecure/placeholder/short keys and succeeds on valid keys
- validate_config triggers explicit validation
- server.py re-exports configuration constants for backward compatibility
"""

import os
import subprocess
import sys
import pytest

from academicai.config import (
    Settings,
    get_settings,
    reload_settings,
    validate_config,
    _validate_proxy_api_key,
    INSECURE_PROXY_KEYS,
)
import server


def test_default_values_when_no_env_vars():
    """Validates that all configuration settings have the expected default values when env is empty."""
    settings = Settings.from_env({})

    assert settings.PORT == 11435
    assert settings.API_KEY == "test-proxy-key-123456"
    assert settings.BASE_URL == "https://academic-ai.boku.ac.at/api/v1"
    assert settings.HEALTH_CHECK_BACKEND is True
    assert settings.ENABLE_COST_MONITORING is True
    assert settings.MAX_MESSAGES == 200
    assert settings.MAX_TOOLS == 64
    assert settings.MAX_MESSAGE_TEXT_CHARS == 500000
    assert settings.MAX_TOOL_SCHEMA_CHARS == 100000
    assert settings.MAX_REQUEST_JSON_CHARS == 2000000
    assert settings.RATE_LIMIT_PER_MINUTE == 120
    assert settings.RATE_LIMIT_WINDOW_SECONDS == 60
    assert settings.ENABLE_HUMANIZATION_PASS is False
    assert settings.HUMANIZATION_TEMPERATURE == 0.7
    assert settings.DEFAULT_CHAT_TEMPERATURE == 0.6
    assert settings.DEFAULT_TOOL_TEMPERATURE == 0.1
    assert settings.STREAM_CHUNK_DELAY_MS == 0
    assert settings.DEBUG_DUMPS is False
    assert settings.ALLOWED_MODELS == ["gpt-4o", "gpt-4o-mini", "gpt-5", "gpt-5-mini"]


def test_custom_env_vars_override_defaults():
    """Validates that custom environment variables override all defaults with correct types."""
    custom_env = {
        "ACADEMICAI_PROXY_PORT": "12345",
        "ACADEMICAI_PROXY_API_KEY": "custom-secure-proxy-key-99999",
        "ACADEMICAI_BASE_URL": "https://custom.academicai.ac.at/v1",
        "ACADEMICAI_HEALTH_CHECK_BACKEND": "false",
        "ACADEMICAI_ENABLE_COST_MONITORING": "false",
        "ACADEMICAI_MAX_MESSAGES": "150",
        "ACADEMICAI_MAX_TOOLS": "32",
        "ACADEMICAI_MAX_MESSAGE_TEXT_CHARS": "100000",
        "ACADEMICAI_MAX_TOOL_SCHEMA_CHARS": "50000",
        "ACADEMICAI_MAX_REQUEST_JSON_CHARS": "1000000",
        "ACADEMICAI_RATE_LIMIT_PER_MINUTE": "60",
        "ACADEMICAI_RATE_LIMIT_WINDOW_SECONDS": "30",
        "ACADEMICAI_ENABLE_HUMANIZATION_PASS": "true",
        "ACADEMICAI_HUMANIZATION_TEMPERATURE": "0.3",
        "ACADEMICAI_DEFAULT_CHAT_TEMPERATURE": "0.8",
        "ACADEMICAI_DEFAULT_TOOL_TEMPERATURE": "0.2",
        "ACADEMICAI_STREAM_CHUNK_DELAY_MS": "25",
        "ACADEMICAI_DEBUG_DUMPS": "true",
        "ACADEMICAI_ALLOWED_MODELS": "gpt-4o,gpt-custom-test",
    }
    settings = Settings.from_env(custom_env)

    assert settings.PORT == 12345
    assert settings.API_KEY == "custom-secure-proxy-key-99999"
    assert settings.BASE_URL == "https://custom.academicai.ac.at/v1"
    assert settings.HEALTH_CHECK_BACKEND is False
    assert settings.ENABLE_COST_MONITORING is False
    assert settings.MAX_MESSAGES == 150
    assert settings.MAX_TOOLS == 32
    assert settings.MAX_MESSAGE_TEXT_CHARS == 100000
    assert settings.MAX_TOOL_SCHEMA_CHARS == 50000
    assert settings.MAX_REQUEST_JSON_CHARS == 1000000
    assert settings.RATE_LIMIT_PER_MINUTE == 60
    assert settings.RATE_LIMIT_WINDOW_SECONDS == 30
    assert settings.ENABLE_HUMANIZATION_PASS is True
    assert settings.HUMANIZATION_TEMPERATURE == 0.3
    assert settings.DEFAULT_CHAT_TEMPERATURE == 0.8
    assert settings.DEFAULT_TOOL_TEMPERATURE == 0.2
    assert settings.STREAM_CHUNK_DELAY_MS == 25
    assert settings.DEBUG_DUMPS is True
    assert settings.ALLOWED_MODELS == ["gpt-4o", "gpt-custom-test"]


def test_safe_import_without_env_vars(tmp_path):
    """
    Validates that importing academicai.config or server does NOT crash with
    RuntimeError at module import time when ACADEMICAI environment variables are unset
    or set to insecure placeholder values.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Test 1: Even when ACADEMICAI_PROXY_API_KEY is explicitly set to an insecure placeholder,
    # importing academicai.config and server must NOT raise RuntimeError at import time.
    code_insecure = (
        "import os\n"
        "os.environ['ACADEMICAI_PROXY_API_KEY'] = 'academicai-proxy'\n"
        "import academicai.config as config\n"
        "assert config.API_KEY == 'academicai-proxy'\n"
        "import server\n"
        "assert server.API_KEY == 'academicai-proxy'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code_insecure],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": repo_root},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import crashed with insecure key: {result.stderr}"

    # Test 2: In an isolated directory without a .env file, when no ACADEMICAI_ env vars exist,
    # importing academicai.config and server must NOT crash and must use defaults.
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("ACADEMICAI_")}
    clean_env["PYTHONPATH"] = repo_root

    code_defaults = (
        "import academicai.config as config\n"
        "assert config.PORT == 11435\n"
        "assert config.API_KEY == 'test-proxy-key-123456'\n"
        "import server\n"
        "assert server.PORT == 11435\n"
        "assert server.API_KEY == 'test-proxy-key-123456'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code_defaults],
        cwd=str(tmp_path),
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import crashed without env vars: {result.stderr}"


def test_validate_proxy_api_key_rejects_placeholders():
    """Validates that insecure placeholder keys raise RuntimeError."""
    test_placeholders = [
        "academicai-proxy",
        "sk-xxx",
        "change-me",
        "changeme",
        "your-secret-key-here",
        "replace-with-strong-key",
    ]
    for p in test_placeholders:
        with pytest.raises(RuntimeError, match="insecure placeholder"):
            _validate_proxy_api_key(p)
        with pytest.raises(RuntimeError, match="insecure placeholder"):
            server._validate_proxy_api_key(p)


def test_validate_proxy_api_key_rejects_short_and_empty():
    """Validates that empty or short keys (< 16 chars) raise RuntimeError."""
    short_keys = ["", "   ", "short-key", "123456789012345"]
    for k in short_keys:
        with pytest.raises(RuntimeError):
            _validate_proxy_api_key(k)
        with pytest.raises(RuntimeError):
            server._validate_proxy_api_key(k)


def test_validate_proxy_api_key_accepts_valid():
    """Validates that valid keys (>= 16 chars, not in insecure set) succeed."""
    valid_key = "valid-strong-test-key-123456"
    assert _validate_proxy_api_key(valid_key) == valid_key
    assert server._validate_proxy_api_key(valid_key) == valid_key
    assert _validate_proxy_api_key(f"  {valid_key}  ") == valid_key


def test_validate_config_explicit_execution():
    """Validates that validate_config runs explicitly and raises only when key is insecure."""
    valid_settings = Settings.from_env({"ACADEMICAI_PROXY_API_KEY": "valid-strong-key-123456"})
    validate_config(valid_settings)

    insecure_settings = Settings.from_env({"ACADEMICAI_PROXY_API_KEY": "academicai-proxy"})
    with pytest.raises(RuntimeError, match="insecure placeholder"):
        validate_config(insecure_settings)

    short_settings = Settings.from_env({"ACADEMICAI_PROXY_API_KEY": "short-key"})
    with pytest.raises(RuntimeError, match="too short"):
        validate_config(short_settings)


def test_server_backward_compatibility_reexports():
    """Validates that server.py re-exports configuration constants and functions."""
    assert hasattr(server, "PORT")
    assert hasattr(server, "API_KEY")
    assert hasattr(server, "BASE_URL")
    assert hasattr(server, "HEALTH_CHECK_BACKEND")
    assert hasattr(server, "ENABLE_COST_MONITORING")
    assert hasattr(server, "MAX_MESSAGES")
    assert hasattr(server, "MAX_TOOLS")
    assert hasattr(server, "MAX_MESSAGE_TEXT_CHARS")
    assert hasattr(server, "MAX_TOOL_SCHEMA_CHARS")
    assert hasattr(server, "MAX_REQUEST_JSON_CHARS")
    assert hasattr(server, "RATE_LIMIT_PER_MINUTE")
    assert hasattr(server, "RATE_LIMIT_WINDOW_SECONDS")
    assert hasattr(server, "ENABLE_HUMANIZATION_PASS")
    assert hasattr(server, "HUMANIZATION_TEMPERATURE")
    assert hasattr(server, "DEFAULT_CHAT_TEMPERATURE")
    assert hasattr(server, "DEFAULT_TOOL_TEMPERATURE")
    assert hasattr(server, "STREAM_CHUNK_DELAY_MS")
    assert hasattr(server, "DEBUG_DUMPS")
    assert hasattr(server, "ALLOWED_MODELS")
    assert hasattr(server, "_validate_proxy_api_key")
    assert callable(server._validate_proxy_api_key)
