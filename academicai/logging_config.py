"""
AcademicAI Logging Configuration Module.

Kapselt:
- Initialisierung und Konfiguration von Root- und Proxy-Loggern
- TimedRotatingFileHandler für reguläre Logs (INFO) und Fehler-Logs (ERROR) mit täglicher Rotation
- StreamHandler für sys.stdout Konsole
- Uvicorn-Logger-Wiring (uvicorn, uvicorn.error, uvicorn.access) mit propagate = False
- Dynamische Pfad- und Einstellungsauflösung über server / academicai.config
- Sauberes Schließen und Ersetzen von Handlern (verhindert Datei-Locks unter Windows)
"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Any, Optional, Union

import academicai.config as config

LOG_FORMAT: str = "%(asctime)s %(levelname)s %(message)s"
log_formatter: logging.Formatter = logging.Formatter(LOG_FORMAT)

info_handler: Optional[TimedRotatingFileHandler] = None
error_handler: Optional[TimedRotatingFileHandler] = None
console_handler: Optional[logging.StreamHandler] = None
root_logger: logging.Logger = logging.getLogger()
log: logging.Logger = logging.getLogger("academicai-proxy")


def get_logger(name: str = "academicai-proxy") -> logging.Logger:
    """Returns a logger with the given name (defaults to 'academicai-proxy')."""
    return logging.getLogger(name)


def _get_setting(name: str, explicit_value: Optional[Any] = None) -> Any:
    """
    Liefert eine Konfigurationseinstellung.
    Priorität:
    1. Explizit übergebener Parameter (falls ungleich None)
    2. Dynamisch gesetztes Attribut auf server (z.B. monkeypatch in Tests)
    3. Modul academicai.config (falls vorhanden)
    """
    if explicit_value is not None:
        return explicit_value

    server_mod = sys.modules.get("server")
    if server_mod is not None and hasattr(server_mod, name):
        val = getattr(server_mod, name)
        if val is not None:
            return val

    if hasattr(config, name):
        val = getattr(config, name)
        if val is not None:
            return val

    return None


def close_handlers() -> None:
    """Schließt aktive File-Handler und entfernt sie von Root- und Uvicorn-Loggern (wichtig für Windows)."""
    global info_handler, error_handler, console_handler, root_logger
    for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        try:
            ulog = logging.getLogger(uvicorn_logger_name)
            ulog.handlers = []
        except Exception:
            pass
    for h in (info_handler, error_handler, console_handler):
        if h is not None:
            try:
                if root_logger is not None:
                    root_logger.removeHandler(h)
                h.close()
            except Exception:
                pass


def configure_logging(
    log_file: Optional[Union[str, Path]] = None,
    err_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Konfiguriert das Logging-System für AcademicAI-Proxy.

    Args:
        log_file: Pfad zur allgemeinen Logdatei (Default: aus LOG_FILE_PATH oder 'server.log')
        err_file: Pfad zur Fehler-Logdatei (Default: aus ERR_FILE_PATH oder 'server.err.log')
        level: Minimum Logging-Level für Root- und Info-Handler (Default: logging.INFO)

    Returns:
        log: Der konfigurierte Proxy-Logger ("academicai-proxy")
    """
    global log_formatter, info_handler, error_handler, console_handler, root_logger, log

    # Vorherige Handler sauber aushängen und schließen
    close_handlers()

    resolved_log_file = _get_setting("LOG_FILE_PATH", log_file)
    if not resolved_log_file:
        resolved_log_file = _get_setting("log_file_path") or "server.log"

    resolved_err_file = _get_setting("ERR_FILE_PATH", err_file)
    if not resolved_err_file:
        resolved_err_file = _get_setting("err_file_path") or "server.err.log"

    log_path = Path(resolved_log_file)
    err_path = Path(resolved_err_file)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)

    log_formatter = logging.Formatter(LOG_FORMAT)

    info_h = TimedRotatingFileHandler(
        str(log_path),
        when="D",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    info_h.setLevel(level)
    info_h.setFormatter(log_formatter)

    error_h = TimedRotatingFileHandler(
        str(err_path),
        when="D",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    error_h.setLevel(logging.ERROR)
    error_h.setFormatter(log_formatter)

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(level)
    console_h.setFormatter(log_formatter)

    info_handler = info_h
    error_handler = error_h
    console_handler = console_h

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(info_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(console_handler)

    for uvicorn_logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        ulog = logging.getLogger(uvicorn_logger_name)
        ulog.handlers = []
        ulog.addHandler(info_handler)
        ulog.addHandler(error_handler)
        ulog.addHandler(console_handler)
        ulog.propagate = False

    log = logging.getLogger("academicai-proxy")

    server_mod = sys.modules.get("server")
    if server_mod is not None:
        server_mod.info_handler = info_handler
        server_mod.error_handler = error_handler
        server_mod.console_handler = console_handler
        server_mod.root_logger = root_logger
        server_mod.log_formatter = log_formatter
        server_mod.log = log
        server_mod.log_file_path = str(log_path)
        server_mod.err_file_path = str(err_path)

    return log


# Initial default logging configuration
configure_logging()

__all__ = [
    "configure_logging",
    "get_logger",
    "close_handlers",
    "log",
    "log_formatter",
    "info_handler",
    "error_handler",
    "console_handler",
    "root_logger",
    "LOG_FORMAT",
]
