from __future__ import annotations

import logging
import os
from pathlib import Path


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(level_icon)s %(name)s %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SUCCESS_HINTS = (
    "completed",
    "created",
    "acquired",
    "started",
    "ensured",
    "allowed",
    "modified",
    "written",
    "moved_to_breakeven",
)


class VisualLogFormatter(logging.Formatter):
    """Add quick visual markers to log records based on severity and message outcome."""

    LEVEL_ICONS = {
        logging.DEBUG: "🔍",
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🔥",
    }

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage().lower()
        level_icon = self.LEVEL_ICONS.get(record.levelno, "•")
        if record.levelno == logging.INFO and any(hint in message for hint in SUCCESS_HINTS):
            level_icon = "✅"
        record.level_icon = level_icon
        return super().format(record)


def setup_logging() -> None:
    """Configure the process-wide logging handlers once."""
    if getattr(setup_logging, "_configured", False):
        return

    log_level = str(os.getenv("LOG_LEVEL", "INFO")).upper()
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_file = os.getenv("LOG_FILE")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    formatter = VisualLogFormatter(DEFAULT_LOG_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        handlers=handlers,
        force=False,
    )
    setup_logging._configured = True
