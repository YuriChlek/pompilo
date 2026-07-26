from __future__ import annotations

import os
from dataclasses import dataclass

from bot_platform_service.config.database_config import get_database_url


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    """Typed PostgreSQL settings for Bot Platform runtime."""

    url: str


@dataclass(frozen=True, slots=True)
class HttpSettings:
    """Typed HTTP settings for the Bot Platform runtime API."""

    host: str
    port: int


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Typed process-level runtime settings."""

    runner_enabled: bool
    runner_poll_interval_seconds: float


@dataclass(frozen=True, slots=True)
class MarketDataSettings:
    """Typed Market Data Service integration settings."""

    base_url: str
    source: str


@dataclass(frozen=True, slots=True)
class SignalEventSettings:
    """Typed downstream signal event publisher settings."""

    enabled: bool
    redis_url: str
    stream_name: str
    max_retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class MarketDataEventSettings:
    """Typed Market Data event consumer settings."""

    enabled: bool
    redis_url: str
    stream_name: str
    consumer_group: str
    consumer_name: str
    read_count: int
    block_milliseconds: int
    retry_backoff_seconds: float


@dataclass(frozen=True, slots=True)
class EventRetentionSettings:
    """Typed retention settings for service-owned event processing records."""

    idempotency_retention_days: int
    audit_retention_days: int


@dataclass(frozen=True, slots=True)
class BotPlatformSettings:
    """Validated service settings parsed once at startup."""

    database: DatabaseSettings
    http: HttpSettings
    runtime: RuntimeSettings
    market_data: MarketDataSettings
    signal_events: SignalEventSettings
    market_data_events: MarketDataEventSettings
    event_retention: EventRetentionSettings

    @classmethod
    def from_env(cls) -> "BotPlatformSettings":
        """Build service settings from environment variables."""

        return cls(
            database=DatabaseSettings(url=get_database_url()),
            http=HttpSettings(
                host=os.getenv("BOT_PLATFORM_HTTP_HOST", "0.0.0.0"),
                port=_parse_port(os.getenv("BOT_PLATFORM_HTTP_PORT", "8092")),
            ),
            runtime=RuntimeSettings(
                runner_enabled=_parse_bool(os.getenv("BOT_PLATFORM_RUNNER_ENABLED", "false")),
                runner_poll_interval_seconds=_parse_positive_float(
                    os.getenv("BOT_PLATFORM_RUNNER_POLL_INTERVAL_SECONDS", "15")
                ),
            ),
            market_data=MarketDataSettings(
                base_url=_parse_base_url(os.getenv("BOT_PLATFORM_MARKET_DATA_BASE_URL", "http://market_data:8010")),
                source=os.getenv("BOT_PLATFORM_MARKET_DATA_SOURCE", "BINANCE_SPOT").strip() or "BINANCE_SPOT",
            ),
            signal_events=SignalEventSettings(
                enabled=_parse_bool(os.getenv("BOT_PLATFORM_SIGNAL_EVENTS_ENABLED", "false")),
                redis_url=_parse_redis_url(
                    os.getenv(
                        "BOT_PLATFORM_SIGNAL_EVENTS_REDIS_URL",
                        os.getenv("REDIS_URL", "redis://redis:6379/0"),
                    )
                ),
                stream_name=_parse_non_empty(os.getenv("BOT_PLATFORM_SIGNAL_EVENTS_STREAM", "bot-platform-signals")),
                max_retries=_parse_non_negative_int(os.getenv("BOT_PLATFORM_SIGNAL_EVENTS_MAX_RETRIES", "3")),
                retry_backoff_seconds=_parse_non_negative_float(
                    os.getenv("BOT_PLATFORM_SIGNAL_EVENTS_RETRY_BACKOFF_SECONDS", "0.25")
                ),
            ),
            market_data_events=MarketDataEventSettings(
                enabled=_parse_bool(os.getenv("BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED", "false")),
                redis_url=_parse_redis_url(
                    os.getenv(
                        "BOT_PLATFORM_MARKET_DATA_EVENTS_REDIS_URL",
                        os.getenv("REDIS_URL", "redis://redis:6379/0"),
                    ),
                    env_name="BOT_PLATFORM_MARKET_DATA_EVENTS_REDIS_URL",
                ),
                stream_name=_parse_non_empty(
                    os.getenv("BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM", "market-data-events"),
                    env_name="BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM",
                ),
                consumer_group=_parse_non_empty(
                    os.getenv("BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_GROUP", "bot-platform"),
                    env_name="BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_GROUP",
                ),
                consumer_name=_parse_non_empty(
                    os.getenv("BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_NAME", "bot-platform-runner"),
                    env_name="BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_NAME",
                ),
                read_count=_parse_positive_int(
                    os.getenv("BOT_PLATFORM_MARKET_DATA_EVENTS_READ_COUNT", "10"),
                    env_name="BOT_PLATFORM_MARKET_DATA_EVENTS_READ_COUNT",
                ),
                block_milliseconds=_parse_non_negative_int(
                    os.getenv("BOT_PLATFORM_MARKET_DATA_EVENTS_BLOCK_MILLISECONDS", "5000"),
                    env_name="BOT_PLATFORM_MARKET_DATA_EVENTS_BLOCK_MILLISECONDS",
                ),
                retry_backoff_seconds=_parse_non_negative_float(
                    os.getenv("BOT_PLATFORM_MARKET_DATA_EVENTS_RETRY_BACKOFF_SECONDS", "1"),
                    env_name="BOT_PLATFORM_MARKET_DATA_EVENTS_RETRY_BACKOFF_SECONDS",
                ),
            ),
            event_retention=EventRetentionSettings(
                idempotency_retention_days=_parse_positive_int(
                    os.getenv("BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS", "5"),
                    env_name="BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS",
                ),
                audit_retention_days=_parse_positive_int(
                    os.getenv("BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS", "5"),
                    env_name="BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS",
                ),
            ),
        )


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError("BOT_PLATFORM_HTTP_PORT must be an integer") from exc
    if port < 1 or port > 65535:
        raise ValueError("BOT_PLATFORM_HTTP_PORT must be between 1 and 65535")
    return port


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("BOT_PLATFORM_RUNNER_ENABLED must be a boolean value")


def _parse_positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("BOT_PLATFORM_RUNNER_POLL_INTERVAL_SECONDS must be a number") from exc
    if parsed <= 0:
        raise ValueError("BOT_PLATFORM_RUNNER_POLL_INTERVAL_SECONDS must be greater than zero")
    return parsed


def _parse_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("BOT_PLATFORM_MARKET_DATA_BASE_URL must start with http:// or https://")
    return normalized


def _parse_redis_url(value: str, *, env_name: str = "BOT_PLATFORM_SIGNAL_EVENTS_REDIS_URL") -> str:
    normalized = value.strip()
    if not normalized.startswith(("redis://", "rediss://")):
        raise ValueError(f"{env_name} must start with redis:// or rediss://")
    return normalized


def _parse_non_empty(value: str, *, env_name: str = "BOT_PLATFORM_SIGNAL_EVENTS_STREAM") -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{env_name} must not be empty")
    return normalized


def _parse_positive_int(value: str, *, env_name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{env_name} must be greater than zero")
    return parsed


def _parse_non_negative_int(value: str, *, env_name: str = "BOT_PLATFORM_SIGNAL_EVENTS_MAX_RETRIES") -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer") from exc
    if parsed < 0:
        raise ValueError(f"{env_name} must be non-negative")
    return parsed


def _parse_non_negative_float(
    value: str,
    *,
    env_name: str = "BOT_PLATFORM_SIGNAL_EVENTS_RETRY_BACKOFF_SECONDS",
) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a number") from exc
    if parsed < 0:
        raise ValueError(f"{env_name} must be non-negative")
    return parsed
