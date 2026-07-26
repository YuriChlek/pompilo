from __future__ import annotations

from enum import StrEnum


class BotMode(StrEnum):
    """Supported signal-only runtime modes."""

    DRY_RUN = "dry_run"
    NOTIFICATION_ONLY = "notification_only"
    SIGNAL_ONLY = "signal_only"


class BotModuleStatus(StrEnum):
    """Registry status for a bot module."""

    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    DEPRECATED = "DEPRECATED"


class BotInstanceStatus(StrEnum):
    """Lifecycle status for one configured bot instance."""

    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    ENABLED = "ENABLED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"


class BotRunStatus(StrEnum):
    """Status for one bot run."""

    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BotTriggerType(StrEnum):
    """Source that triggered a bot run."""

    MANUAL = "manual"
    SCHEDULER = "scheduler"
    EVENT = "event"


class BotSignalType(StrEnum):
    """Normalized signal categories."""

    ENTRY = "entry"
    EXIT = "exit"
    REBALANCE = "rebalance"
    HOLD = "hold"
    ALERT = "alert"


class BotSignalSide(StrEnum):
    """Optional normalized trade side for a signal."""

    BUY = "buy"
    SELL = "sell"


class BotSignalPublishStatus(StrEnum):
    """Persistence status for a signal."""

    CREATED = "CREATED"
    PUBLISHED = "PUBLISHED"
    IGNORED = "IGNORED"


class BotHealthStatus(StrEnum):
    """Health status reported by a bot module or instance."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class BotPermission(StrEnum):
    """Per-instance runtime permissions."""

    READ_MARKET_DATA = "read_market_data"
    SEND_NOTIFICATIONS = "send_notifications"
    PUBLISH_SIGNALS = "publish_signals"
    READ_STATE = "read_state"
    WRITE_STATE = "write_state"
    EMIT_AUDIT_EVENTS = "emit_audit_events"


class BotNotificationStatus(StrEnum):
    """Outcome of a notification attempt."""

    SKIPPED = "SKIPPED"
    SENT = "SENT"
    FAILED = "FAILED"


class BotStateChangeOperation(StrEnum):
    """State mutation requested by a bot adapter."""

    UPSERT = "UPSERT"
    DELETE = "DELETE"
