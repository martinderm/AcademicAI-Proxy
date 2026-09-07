"""
Unit tests for academicai.runtime lifecycle helpers.

Tests:
- write_pid_file and cleanup_pid_file:
  - writes current PID and ensures parent directory creation (tmp_path)
  - cleans up PID file when PID matches
  - preserves PID file when PID does not match
  - handles missing PID file gracefully
  - dynamic lookup of PID_FILE on server module
  - exception handling during write and cleanup
- check_backend_health:
  - disabled via parameter (enabled=False)
  - disabled via server setting / config
  - successful check with 200 OK (latency, status_code, ok=True)
  - failed check with non-200 (status_code, ok=False, error message)
  - exception handling during request (ok=False, error message)
- get_health_payload:
  - status 'ok' when backend healthy or disabled
  - status 'degraded' when backend is unhealthy
  - automatic fallback to check_backend_health() when called with no arguments
- server backward compatibility:
  - server re-exports all required runtime symbols
"""

import os
from pathlib import Path
from typing import Any
import pytest
import httpx

import server
from academicai.runtime import (
    write_pid_file,
    _write_pid_file,
    cleanup_pid_file,
    _cleanup_pid_file,
    check_backend_health,
    _check_backend_health,
    get_health_payload,
)


# ---------------------------------------------------------------------------
# 1. PID File Handling
# ---------------------------------------------------------------------------


def test_write_pid_file_creates_file_and_parent_dirs(tmp_path: Path):
    target = tmp_path / "nested" / "deep" / "server.pid"
    assert not target.exists()

    write_pid_file(target)

    assert target.exists()
    content = target.read_text(encoding="utf-8").strip()
    assert content == str(os.getpid())


def test_cleanup_pid_file_matching_pid(tmp_path: Path):
    target = tmp_path / "server.pid"
    write_pid_file(target)
    assert target.exists()

    cleanup_pid_file(target)
    assert not target.exists()


def test_cleanup_pid_file_non_matching_pid(tmp_path: Path):
    target = tmp_path / "server.pid"
    target.write_text("999999999\n", encoding="utf-8")
    assert target.exists()

    cleanup_pid_file(target)
    assert target.exists()
    assert target.read_text(encoding="utf-8").strip() == "999999999"


def test_cleanup_pid_file_missing_file(tmp_path: Path):
    target = tmp_path / "nonexistent.pid"
    assert not target.exists()

    # Should not raise exception
    cleanup_pid_file(target)
    assert not target.exists()


def test_pid_file_dynamic_lookup_from_server(monkeypatch, tmp_path: Path):
    target = tmp_path / "server_dyn.pid"
    monkeypatch.setattr(server, "PID_FILE", target)

    # Calling without arguments uses server.PID_FILE
    write_pid_file()
    assert target.exists()
    assert target.read_text(encoding="utf-8").strip() == str(os.getpid())

    cleanup_pid_file()
    assert not target.exists()


def test_pid_file_handles_exceptions_gracefully(monkeypatch, tmp_path: Path):
    class BadPath:
        @property
        def parent(self):
            raise OSError("disk error")

    # Should log warning but not raise unhandled exception
    write_pid_file(BadPath())  # type: ignore
    cleanup_pid_file(BadPath())  # type: ignore


# ---------------------------------------------------------------------------
# 2. check_backend_health
# ---------------------------------------------------------------------------


def test_check_backend_health_disabled_argument():
    result = check_backend_health(enabled=False)
    assert result == {"enabled": False, "ok": None}


def test_check_backend_health_disabled_server_setting(monkeypatch):
    monkeypatch.setattr(server, "HEALTH_CHECK_BACKEND", False)
    result = check_backend_health()
    assert result == {"enabled": False, "ok": None}


class _MockResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _MockClient:
    def __init__(self, response: Any = None, exc: Exception = None):
        self._response = response
        self._exc = exc

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get(self, url, headers=None):
        if self._exc:
            raise self._exc
        return self._response


def test_check_backend_health_200_ok(monkeypatch):
    mock_client = _MockClient(response=_MockResponse(200))
    monkeypatch.setattr("academicai.runtime.httpx.Client", lambda **kwargs: mock_client)

    result = check_backend_health(
        enabled=True,
        timeout=1.0,
        base_url="https://test.academicai.ac.at",
        headers={"X-Client-ID": "test"},
    )
    assert result["enabled"] is True
    assert result["ok"] is True
    assert result["status_code"] == 200
    assert "latency_ms" in result
    assert result["latency_ms"] >= 0
    assert "error" not in result


def test_check_backend_health_non_200(monkeypatch):
    mock_client = _MockClient(response=_MockResponse(502))
    monkeypatch.setattr("academicai.runtime.httpx.Client", lambda **kwargs: mock_client)

    result = check_backend_health(
        enabled=True,
        base_url="https://test.academicai.ac.at",
    )
    assert result["enabled"] is True
    assert result["ok"] is False
    assert result["status_code"] == 502
    assert result["error"] == "backend responded with non-200 status"
    assert "latency_ms" in result


def test_check_backend_health_exception(monkeypatch):
    mock_client = _MockClient(exc=httpx.ConnectTimeout("connection timed out"))
    monkeypatch.setattr("academicai.runtime.httpx.Client", lambda **kwargs: mock_client)

    result = check_backend_health(
        enabled=True,
        base_url="https://test.academicai.ac.at",
    )
    assert result["enabled"] is True
    assert result["ok"] is False
    assert "timed out" in result["error"]
    assert "latency_ms" in result


def test_check_backend_health_direct_client_injection():
    mock_client = _MockClient(response=_MockResponse(200))
    result = check_backend_health(
        enabled=True,
        client=mock_client,  # type: ignore
        base_url="https://test.academicai.ac.at",
    )
    assert result["enabled"] is True
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# 3. get_health_payload
# ---------------------------------------------------------------------------


def test_get_health_payload_ok():
    backend_data = {"enabled": True, "ok": True, "latency_ms": 15, "status_code": 200}
    payload = get_health_payload(backend_data)
    assert payload == {
        "status": "ok",
        "service": "academicai-proxy",
        "backend": backend_data,
    }


def test_get_health_payload_disabled_backend():
    backend_data = {"enabled": False, "ok": None}
    payload = get_health_payload(backend_data)
    assert payload == {
        "status": "ok",
        "service": "academicai-proxy",
        "backend": backend_data,
    }


def test_get_health_payload_degraded_when_backend_unhealthy():
    backend_data = {
        "enabled": True,
        "ok": False,
        "latency_ms": 105,
        "error": "backend responded with non-200 status",
        "status_code": 500,
    }
    payload = get_health_payload(backend_data)
    assert payload == {
        "status": "degraded",
        "service": "academicai-proxy",
        "backend": backend_data,
    }


def test_get_health_payload_no_args_calls_check_backend(monkeypatch):
    monkeypatch.setattr("academicai.runtime.check_backend_health", lambda: {"enabled": True, "ok": True, "latency_ms": 8})
    payload = get_health_payload()
    assert payload["status"] == "ok"
    assert payload["backend"]["ok"] is True


# ---------------------------------------------------------------------------
# 4. Server re-exports backward compatibility
# ---------------------------------------------------------------------------


def test_server_reexports_runtime_symbols():
    symbols = [
        "write_pid_file",
        "_write_pid_file",
        "cleanup_pid_file",
        "_cleanup_pid_file",
        "check_backend_health",
        "_check_backend_health",
        "get_health_payload",
    ]
    for sym in symbols:
        assert hasattr(server, sym), f"server is missing expected re-export: {sym}"
        assert callable(getattr(server, sym))
