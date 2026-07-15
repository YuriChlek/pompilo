from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class StructuredLogRecord:
    """JSON-serializable structured log payload."""

    level: str
    event: str
    fields: dict[str, object]

    def to_json(self) -> str:
        """Render the log record as stable JSON."""

        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": self.level,
            "event": self.event,
            **self.fields,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


class JsonStructuredLogger:
    """StructuredLogger implementation backed by stdlib logging."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("bot_platform_service")

    def info(self, event: str, **fields: object) -> None:
        """Record an informational structured event."""

        self.logger.info(StructuredLogRecord("INFO", event, dict(fields)).to_json())

    def warning(self, event: str, **fields: object) -> None:
        """Record a warning structured event."""

        self.logger.warning(StructuredLogRecord("WARNING", event, dict(fields)).to_json())

    def error(self, event: str, **fields: object) -> None:
        """Record an error structured event."""

        self.logger.error(StructuredLogRecord("ERROR", event, dict(fields)).to_json())


__all__ = ["JsonStructuredLogger", "StructuredLogRecord"]
