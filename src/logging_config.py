"""Structured logging configuration for the ZeroG-RL system.

Uses ``structlog`` for structured, machine-parseable logging. Supports both
human-readable console output (development) and JSON output (production).
"""

from __future__ import annotations

import atexit
import logging
import sys
from pathlib import Path
from typing import IO

import structlog

# Module-level reference so the file handle survives for the process lifetime
# and can be cleaned up via atexit.
_log_file_handle: IO[str] | None = None


def _close_log_file() -> None:
    """Close the log file handle on interpreter shutdown."""
    global _log_file_handle
    if _log_file_handle is not None:
        _log_file_handle.close()
        _log_file_handle = None


def setup_logging(
    log_level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
) -> None:
    """Configure structured logging for the entire application.

    This should be called once at application startup, before any logging
    statements are executed.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write logs to a file. When ``None``, logs
            are written to ``stdout``.
        json_format: When ``True``, emit JSON-formatted log lines (useful in
            production / container environments).
    """
    global _log_file_handle

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format or log_file is not None:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    all_processors = [*shared_processors, renderer]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _log_file_handle = log_file.open("a")
        atexit.register(_close_log_file)
        factory: structlog.types.WrappedLogger = structlog.WriteLoggerFactory(
            file=_log_file_handle,
        )
    else:
        factory = structlog.PrintLoggerFactory(file=sys.stdout)

    structlog.configure(
        processors=all_processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=factory,
        cache_logger_on_first_use=True,
    )
