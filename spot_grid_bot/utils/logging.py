from __future__ import annotations

import logging
import sys


class _LevelColorFormatter(logging.Formatter):
    """Apply ANSI colors only to warning and error log levels."""

    WARNING_COLOR = "\033[33m"
    ERROR_COLOR = "\033[31m"
    RESET_COLOR = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format one record and colorize it based on severity."""
        message = super().format(record)
        if record.levelno == logging.WARNING:
            return f"{self.WARNING_COLOR}{message}{self.RESET_COLOR}"
        if record.levelno >= logging.ERROR:
            return f"{self.ERROR_COLOR}{message}{self.RESET_COLOR}"
        return message


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the default project logging format and severity level."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LevelColorFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
