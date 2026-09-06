"""
Unit tests for academicai.logging_config module and server logging delegation.

Tests:
- configure_logging sets up root logger, handlers, formatters, levels, and uvicorn wiring
- uvicorn loggers (uvicorn, uvicorn.error, uvicorn.access) have propagate=False and proper handlers
- configure_logging with custom paths (e.g. in tmp_path) and levels works without error
- file logging separates INFO and ERROR messages properly
- get_logger helper provides proxy and custom named loggers
- dynamic lookup of LOG_FILE_PATH / ERR_FILE_PATH from server module
- close_handlers cleans up active handlers and detaches them
- server module re-exports all required logging symbols for backward compatibility
- calling server.configure_logging maintains synchronization with server attributes
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
import sys
import pytest

import server
import academicai.logging_config as lc
from academicai.logging_config import (
    configure_logging,
    get_logger,
    close_handlers,
    log,
    log_formatter,
    info_handler,
    error_handler,
    console_handler,
    root_logger,
    LOG_FORMAT,
)


@pytest.fixture(autouse=True)
def clean_logging_state(tmp_path: Path):
    """
    Ensure logging handlers pointing to tmp_path or test files are closed
    after each test so Windows does not lock files or directories.
    """
    yield
    lc.close_handlers()
    # Restore default logging for any following tests
    lc.configure_logging()


# ---------------------------------------------------------------------------
# 1. Default Configuration & Wiring
# ---------------------------------------------------------------------------


def test_configure_logging_default_setup():
    proxy_log = configure_logging()

    assert proxy_log.name == "academicai-proxy"
    assert lc.log.name == "academicai-proxy"

    # Root logger
    assert lc.root_logger is not None
    assert lc.root_logger.level == logging.INFO

    # Formatter
    assert lc.log_formatter is not None
    assert lc.log_formatter._fmt == LOG_FORMAT

    # Info handler
    assert isinstance(lc.info_handler, TimedRotatingFileHandler)
    assert lc.info_handler.level == logging.INFO
    assert lc.info_handler.when == "D"
    assert lc.info_handler.backupCount == 30
    assert lc.info_handler.encoding == "utf-8"
    assert lc.info_handler.formatter is lc.log_formatter

    # Error handler
    assert isinstance(lc.error_handler, TimedRotatingFileHandler)
    assert lc.error_handler.level == logging.ERROR
    assert lc.error_handler.when == "D"
    assert lc.error_handler.backupCount == 30
    assert lc.error_handler.encoding == "utf-8"
    assert lc.error_handler.formatter is lc.log_formatter

    # Console handler
    assert isinstance(lc.console_handler, logging.StreamHandler)
    assert lc.console_handler.stream == sys.stdout
    assert lc.console_handler.level == logging.INFO
    assert lc.console_handler.formatter is lc.log_formatter

    # Handlers attached to root logger
    assert lc.info_handler in lc.root_logger.handlers
    assert lc.error_handler in lc.root_logger.handlers
    assert lc.console_handler in lc.root_logger.handlers


def test_uvicorn_loggers_wiring():
    configure_logging()

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        ulog = logging.getLogger(name)
        assert ulog.propagate is False
        assert lc.info_handler in ulog.handlers
        assert lc.error_handler in ulog.handlers
        assert lc.console_handler in ulog.handlers


# ---------------------------------------------------------------------------
# 2. Custom Paths, Levels & Logging Output
# ---------------------------------------------------------------------------


def test_configure_logging_custom_paths_and_levels(tmp_path: Path):
    custom_log = tmp_path / "subfolder" / "custom_app.log"
    custom_err = tmp_path / "subfolder" / "custom_app.err.log"

    # Parent directory should not exist beforehand
    assert not custom_log.parent.exists()

    proxy_log = configure_logging(
        log_file=custom_log,
        err_file=custom_err,
        level=logging.DEBUG,
    )

    # Parent directory created automatically
    assert custom_log.parent.exists()
    assert Path(lc.info_handler.baseFilename).resolve() == custom_log.resolve()
    assert Path(lc.error_handler.baseFilename).resolve() == custom_err.resolve()

    assert lc.root_logger.level == logging.DEBUG
    assert lc.info_handler.level == logging.DEBUG
    assert lc.error_handler.level == logging.ERROR

    # Emit logs
    proxy_log.debug("test-debug-message")
    proxy_log.info("test-info-message")
    proxy_log.error("test-error-message")

    # Flush handlers
    lc.info_handler.flush()
    lc.error_handler.flush()

    info_content = custom_log.read_text(encoding="utf-8")
    err_content = custom_err.read_text(encoding="utf-8")

    # Info file receives DEBUG, INFO, and ERROR
    assert "test-debug-message" in info_content
    assert "test-info-message" in info_content
    assert "test-error-message" in info_content

    # Error file ONLY receives ERROR
    assert "test-error-message" in err_content
    assert "test-info-message" not in err_content
    assert "test-debug-message" not in err_content


def test_configure_logging_dynamic_lookup_from_server(monkeypatch, tmp_path: Path):
    custom_log = tmp_path / "server_dyn.log"
    custom_err = tmp_path / "server_dyn.err.log"

    monkeypatch.setattr(server, "LOG_FILE_PATH", str(custom_log))
    monkeypatch.setattr(server, "ERR_FILE_PATH", str(custom_err))

    configure_logging()

    assert Path(lc.info_handler.baseFilename).resolve() == custom_log.resolve()
    assert Path(lc.error_handler.baseFilename).resolve() == custom_err.resolve()


# ---------------------------------------------------------------------------
# 3. Helpers & Cleanup
# ---------------------------------------------------------------------------


def test_get_logger():
    default_log = get_logger()
    assert default_log.name == "academicai-proxy"

    custom_log = get_logger("my-custom-component")
    assert custom_log.name == "my-custom-component"


def test_close_handlers():
    configure_logging()

    assert len(lc.root_logger.handlers) >= 3
    lc.close_handlers()

    # After close, handlers are detached from root
    assert lc.info_handler not in lc.root_logger.handlers
    assert lc.error_handler not in lc.root_logger.handlers
    assert lc.console_handler not in lc.root_logger.handlers

    # Uvicorn loggers are cleared
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        assert len(logging.getLogger(name).handlers) == 0


# ---------------------------------------------------------------------------
# 4. Server Backward Compatibility & Re-exports
# ---------------------------------------------------------------------------


def test_server_reexports_logging_symbols():
    required_symbols = [
        "log",
        "log_formatter",
        "info_handler",
        "error_handler",
        "console_handler",
        "root_logger",
        "configure_logging",
        "get_logger",
        "close_handlers",
    ]
    for sym in required_symbols:
        assert hasattr(server, sym), f"server is missing expected re-export: {sym}"

    assert isinstance(server.log, logging.Logger)
    assert isinstance(server.log_formatter, logging.Formatter)
    assert isinstance(server.info_handler, TimedRotatingFileHandler)
    assert isinstance(server.error_handler, TimedRotatingFileHandler)
    assert isinstance(server.console_handler, logging.StreamHandler)
    assert isinstance(server.root_logger, logging.Logger)
    assert callable(server.configure_logging)
    assert callable(server.get_logger)


def test_server_configure_logging_synchronization(tmp_path: Path):
    custom_log = tmp_path / "sync.log"
    custom_err = tmp_path / "sync.err.log"

    # Calling configure_logging via server updates server's re-exported handlers
    server.configure_logging(log_file=custom_log, err_file=custom_err)

    assert server.info_handler is lc.info_handler
    assert server.error_handler is lc.error_handler
    assert server.console_handler is lc.console_handler
    assert server.root_logger is lc.root_logger
    assert server.log is lc.log

    assert Path(server.info_handler.baseFilename).resolve() == custom_log.resolve()
    assert Path(server.error_handler.baseFilename).resolve() == custom_err.resolve()
