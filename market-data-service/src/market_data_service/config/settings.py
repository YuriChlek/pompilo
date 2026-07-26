from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
import os
from urllib.parse import quote_plus

from sqlalchemy.engine import make_url

from market_data_service.config.provider_config import BinanceSpotProviderConfig, BybitSpotProviderConfig
from market_data_service.config.queue_config import RedisStreamBrokerConfig, calculate_redis_stream_maxlen
from market_data_service.config.scheduler_config import SchedulerConfig
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.symbol_normalization import normalize_symbol
from market_data_service.domain.scheduler import SUPPORTED_SCHEDULE_TIMEFRAMES


class SettingsError(ValueError):
    """Raised when Market Data Service configuration is invalid."""


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    database_url: str


@dataclass(frozen=True, slots=True)
class HttpSettings:
    host: str
    port: int


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str


@dataclass(frozen=True, slots=True)
class ShutdownSettings:
    timeout_seconds: float


@dataclass(frozen=True, slots=True)
class BackfillSettings:
    batch_size_candles: int
    max_concurrency: int


@dataclass(frozen=True, slots=True)
class AvailabilitySettings:
    availability_ttl_hours: float
    unsupported_recheck_hours: float
    temporary_error_recheck_minutes: float


@dataclass(frozen=True, slots=True)
class RetentionSettings:
    redis_event_retention_days: int
    outbox_retention_days: int


@dataclass(frozen=True, slots=True)
class BootstrapSettings:
    bootstrap_lookback_years: int
    bootstrap_max_chunks_per_tick: int
    provider_max_concurrency: int


@dataclass(frozen=True, slots=True)
class CollectSettings:
    poll_interval_seconds: float
    on_start: bool
    max_jobs_per_tick: int


@dataclass(frozen=True, slots=True)
class MarketDataServiceSettings:
    database: DatabaseSettings
    redis: RedisStreamBrokerConfig
    provider: BinanceSpotProviderConfig
    bybit_provider: BybitSpotProviderConfig
    provider_mode: str
    scheduler: SchedulerConfig
    http: HttpSettings
    logging: LoggingSettings
    shutdown: ShutdownSettings
    backfill: BackfillSettings
    availability: AvailabilitySettings
    retention: RetentionSettings
    bootstrap: BootstrapSettings
    collect: CollectSettings
    provider_priority: tuple[str, ...]
    scheduler_enabled: bool
    outbox_publisher_enabled: bool


def load_settings(env: Mapping[str, str] | None = None) -> MarketDataServiceSettings:
    source = env or os.environ
    return MarketDataServiceSettings(
        database=load_database_settings(source),
        redis=load_redis_settings(source),
        provider=load_provider_settings(source),
        bybit_provider=load_bybit_provider_settings(source),
        provider_mode=load_provider_mode(source),
        scheduler=load_scheduler_settings(source),
        http=load_http_settings(source),
        logging=load_logging_settings(source),
        shutdown=load_shutdown_settings(source),
        backfill=load_backfill_settings(source),
        availability=load_availability_settings(source),
        retention=load_retention_settings(source),
        bootstrap=load_bootstrap_settings(source),
        collect=load_collect_settings(source),
        provider_priority=load_provider_priority(source),
        scheduler_enabled=_parse_bool(source, "MARKET_DATA_SCHEDULER_ENABLED", default=True),
        outbox_publisher_enabled=_parse_bool(source, "MARKET_DATA_OUTBOX_PUBLISHER_ENABLED", default=True),
    )


def load_database_settings(env: Mapping[str, str] | None = None) -> DatabaseSettings:
    source = env or os.environ
    explicit_url = _optional(source, "MARKET_DATA_DATABASE_URL") or _optional(source, "DATABASE_URL")
    database_url = explicit_url or _build_database_url(source)

    try:
        parsed_url = make_url(database_url)
    except Exception as exc:
        raise SettingsError(f"MARKET_DATA_DATABASE_URL/DATABASE_URL is invalid: {exc}") from exc

    if not parsed_url.host:
        raise SettingsError("Database host is required")
    if not parsed_url.database:
        raise SettingsError("Database name is required")
    if not parsed_url.drivername.startswith("postgresql"):
        raise SettingsError("Database URL must use a PostgreSQL driver")

    return DatabaseSettings(database_url=database_url)


def load_redis_settings(env: Mapping[str, str] | None = None) -> RedisStreamBrokerConfig:
    source = env or os.environ
    raw_maxlen = _optional(source, "MARKET_DATA_OUTBOX_STREAM_MAXLEN")
    redis_url = _optional(source, "MARKET_DATA_REDIS_URL") or _optional(source, "REDIS_URL") or "redis://localhost:6379/0"
    stream_name = _optional(source, "MARKET_DATA_OUTBOX_STREAM") or "market-data-events"

    if not redis_url.startswith(("redis://", "rediss://")):
        raise SettingsError("MARKET_DATA_REDIS_URL must start with redis:// or rediss://")
    if not stream_name:
        raise SettingsError("MARKET_DATA_OUTBOX_STREAM must not be empty")

    maxlen = _parse_optional_positive_int(raw_maxlen, "MARKET_DATA_OUTBOX_STREAM_MAXLEN")
    if maxlen is None:
        redis_event_retention_days = _parse_positive_int(source, "MARKET_DATA_REDIS_EVENT_RETENTION_DAYS", default="2")
        maxlen = calculate_redis_stream_maxlen(redis_event_retention_days)

    return RedisStreamBrokerConfig(
        redis_url=redis_url,
        stream_name=stream_name,
        maxlen=maxlen,
    )


def load_provider_settings(env: Mapping[str, str] | None = None) -> BinanceSpotProviderConfig:
    source = env or os.environ
    rest_endpoint = (_optional(source, "BINANCE_REST_ENDPOINT") or "https://api.binance.com").rstrip("/")
    if not rest_endpoint.startswith(("http://", "https://")):
        raise SettingsError("BINANCE_REST_ENDPOINT must start with http:// or https://")

    return BinanceSpotProviderConfig(
        rest_endpoint=rest_endpoint,
        request_timeout_seconds=_parse_positive_float(source, "BINANCE_REQUEST_TIMEOUT_SECONDS", default="30"),
        max_limit=_parse_positive_int(source, "BINANCE_KLINE_LIMIT", default="1000"),
        safety_delay_by_timeframe=_load_safety_delays(source),
        max_concurrent_requests=_parse_positive_int(source, "BINANCE_MAX_CONCURRENT_REQUESTS", default="8"),
        circuit_breaker_failure_threshold=_parse_positive_int(
            source,
            "BINANCE_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            default="5",
        ),
        circuit_breaker_recovery_timeout_seconds=_parse_positive_float(
            source,
            "BINANCE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
            default="60",
        ),
    )


def load_bybit_provider_settings(env: Mapping[str, str] | None = None) -> BybitSpotProviderConfig:
    source = env or os.environ
    rest_endpoint = (_optional(source, "BYBIT_REST_ENDPOINT") or "https://api.bybit.com").rstrip("/")
    if not rest_endpoint.startswith(("http://", "https://")):
        raise SettingsError("BYBIT_REST_ENDPOINT must start with http:// or https://")

    return BybitSpotProviderConfig(
        rest_endpoint=rest_endpoint,
        request_timeout_seconds=_parse_positive_float(source, "BYBIT_REQUEST_TIMEOUT_SECONDS", default="30"),
        max_limit=_parse_positive_int(source, "BYBIT_KLINE_LIMIT", default="1000"),
        safety_delay_by_timeframe=_load_safety_delays(source),
        max_concurrent_requests=_parse_positive_int(source, "BYBIT_MAX_CONCURRENT_REQUESTS", default="8"),
        circuit_breaker_failure_threshold=_parse_positive_int(
            source,
            "BYBIT_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            default="5",
        ),
        circuit_breaker_recovery_timeout_seconds=_parse_positive_float(
            source,
            "BYBIT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
            default="60",
        ),
    )


def load_provider_mode(env: Mapping[str, str] | None = None) -> str:
    source = env or os.environ
    provider_mode = (_optional(source, "MARKET_DATA_PROVIDER_MODE") or "binance").lower()
    if provider_mode not in {"binance", "fixture"}:
        raise SettingsError("MARKET_DATA_PROVIDER_MODE must be one of: binance, fixture")
    return provider_mode


def load_provider_priority(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    source = env or os.environ
    raw_priority = _optional(source, "MARKET_DATA_PROVIDER_PRIORITY")
    if raw_priority is None:
        return ("binance", "bybit")

    providers = _parse_csv(raw_priority, lowercase=True)
    allowed = {"binance", "bybit"}
    unsupported = sorted(set(providers).difference(allowed))
    if unsupported:
        raise SettingsError(f"Unsupported provider in priority: {', '.join(unsupported)}")
    if len(providers) != len(set(providers)):
        raise SettingsError("Duplicate providers in priority list")
    return providers


def load_scheduler_settings(env: Mapping[str, str] | None = None) -> SchedulerConfig:
    source = env or os.environ
    timeframes = _parse_csv(_optional(source, "MARKET_DATA_TIMEFRAMES") or "1h,4h,1d", lowercase=True)
    unsupported = sorted(set(timeframes).difference(SUPPORTED_SCHEDULE_TIMEFRAMES))
    if unsupported:
        raise SettingsError(f"Unsupported scheduler timeframe(s): {', '.join(unsupported)}")

    return SchedulerConfig(
        source=MarketDataSource(_optional(source, "MARKET_DATA_SOURCE") or MarketDataSource.BINANCE_SPOT.value),
        provider_symbols=tuple(
            normalize_symbol(symbol)
            for symbol in _parse_csv(_optional(source, "MARKET_DATA_PROVIDER_SYMBOLS") or "ETHUSDT")
        ),
        timeframes=timeframes,
        safety_delay_by_timeframe=_load_safety_delays(source),
        jitter_seconds=_parse_non_negative_int(source, "MARKET_DATA_SCHEDULER_JITTER_SECONDS", default="30"),
        poll_interval_seconds=_parse_positive_float(
            source,
            "MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS",
            default=_optional(source, "MARKET_DATA_SCHEDULER_POLL_INTERVAL_SECONDS") or "30",
        ),
    )


def load_http_settings(env: Mapping[str, str] | None = None) -> HttpSettings:
    source = env or os.environ
    host = _optional(source, "MARKET_DATA_HTTP_HOST") or "0.0.0.0"
    port = _parse_port(source, "MARKET_DATA_HTTP_PORT", default="8010")
    return HttpSettings(host=host, port=port)


def load_availability_settings(env: Mapping[str, str] | None = None) -> AvailabilitySettings:
    source = env or os.environ
    return AvailabilitySettings(
        availability_ttl_hours=_parse_positive_float(
            source, "MARKET_DATA_PROVIDER_AVAILABILITY_TTL_HOURS", default="24.0"
        ),
        unsupported_recheck_hours=_parse_positive_float(
            source, "MARKET_DATA_UNSUPPORTED_SYMBOL_RECHECK_HOURS", default="24.0"
        ),
        temporary_error_recheck_minutes=_parse_positive_float(
            source, "MARKET_DATA_TEMPORARY_ERROR_RECHECK_MINUTES", default="5.0"
        ),
    )


def load_logging_settings(env: Mapping[str, str] | None = None) -> LoggingSettings:
    source = env or os.environ
    level = (_optional(source, "MARKET_DATA_LOG_LEVEL") or "INFO").upper()
    allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if level not in allowed_levels:
        raise SettingsError(f"MARKET_DATA_LOG_LEVEL must be one of: {', '.join(sorted(allowed_levels))}")
    return LoggingSettings(level=level)


def load_shutdown_settings(env: Mapping[str, str] | None = None) -> ShutdownSettings:
    source = env or os.environ
    return ShutdownSettings(
        timeout_seconds=_parse_positive_float(source, "MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS", default="30")
    )


def load_backfill_settings(env: Mapping[str, str] | None = None) -> BackfillSettings:
    source = env or os.environ
    return BackfillSettings(
        batch_size_candles=_parse_positive_int(source, "MARKET_DATA_BACKFILL_BATCH_CANDLES", default="500"),
        max_concurrency=_parse_positive_int(source, "MARKET_DATA_BACKFILL_MAX_CONCURRENCY", default="1"),
    )


def load_retention_settings(env: Mapping[str, str] | None = None) -> RetentionSettings:
    source = env or os.environ
    return RetentionSettings(
        redis_event_retention_days=_parse_positive_int(source, "MARKET_DATA_REDIS_EVENT_RETENTION_DAYS", default="2"),
        outbox_retention_days=_parse_positive_int(source, "MARKET_DATA_OUTBOX_RETENTION_DAYS", default="5"),
    )


def load_bootstrap_settings(env: Mapping[str, str] | None = None) -> BootstrapSettings:
    source = env or os.environ
    return BootstrapSettings(
        bootstrap_lookback_years=_parse_positive_int(source, "MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS", default="2"),
        bootstrap_max_chunks_per_tick=_parse_positive_int(source, "MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK", default="10"),
        provider_max_concurrency=_parse_positive_int(source, "MARKET_DATA_PROVIDER_MAX_CONCURRENCY", default="8"),
    )


def load_collect_settings(env: Mapping[str, str] | None = None) -> CollectSettings:
    source = env or os.environ
    return CollectSettings(
        poll_interval_seconds=_parse_positive_float(
            source,
            "MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS",
            default=_optional(source, "MARKET_DATA_SCHEDULER_POLL_INTERVAL_SECONDS") or "30",
        ),
        on_start=_parse_bool(source, "MARKET_DATA_COLLECT_ON_START", default=True),
        max_jobs_per_tick=_parse_positive_int(source, "MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK", default="100"),
    )


def _build_database_url(env: Mapping[str, str]) -> str:
    user = _optional(env, "DB_USER") or "admin"
    password = _optional(env, "DB_PASS") or _optional(env, "DB_PASSWORD") or "admin_pass"
    host = _optional(env, "DB_HOST") or "localhost"
    port = _optional(env, "DB_PORT") or "5432"
    database = _optional(env, "DB_NAME") or _optional(env, "DATABASE") or "pompilo_db"
    return (
        "postgresql+asyncpg://"
        f"{quote_plus(user)}:{quote_plus(password)}@{host}:{port}/{quote_plus(database)}"
    )


def _load_safety_delays(env: Mapping[str, str]) -> dict[str, timedelta]:
    return {
        "1h": timedelta(seconds=_parse_non_negative_int(env, "MARKET_DATA_1H_SAFETY_DELAY_SECONDS", default="10")),
        "4h": timedelta(seconds=_parse_non_negative_int(env, "MARKET_DATA_4H_SAFETY_DELAY_SECONDS", default="20")),
        "1d": timedelta(seconds=_parse_non_negative_int(env, "MARKET_DATA_1D_SAFETY_DELAY_SECONDS", default="30")),
    }


def _optional(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_csv(value: str, *, lowercase: bool = False) -> tuple[str, ...]:
    items = tuple(item.strip().lower() if lowercase else item.strip().upper() for item in value.split(",") if item.strip())
    if not items:
        raise SettingsError("CSV environment value must contain at least one item")
    return items


def _parse_optional_positive_int(value: str | None, name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise SettingsError(f"{name} must be positive")
    return parsed


def _parse_non_negative_int(env: Mapping[str, str], name: str, *, default: str) -> int:
    try:
        value = int(_optional(env, name) or default)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a valid integer") from exc
    if value < 0:
        raise SettingsError(f"{name} must be non-negative")
    return value


def _parse_positive_int(env: Mapping[str, str], name: str, *, default: str) -> int:
    try:
        value = int(_optional(env, name) or default)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a valid integer") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be positive")
    return value


def _parse_positive_float(env: Mapping[str, str], name: str, *, default: str) -> float:
    try:
        value = float(_optional(env, name) or default)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a valid float") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be positive")
    return value


def _parse_port(env: Mapping[str, str], name: str, *, default: str) -> int:
    value = _parse_positive_int(env, name, default=default)
    if value > 65535:
        raise SettingsError(f"{name} must be between 1 and 65535")
    return value


def _parse_bool(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw_value = _optional(env, name)
    if raw_value is None:
        return default
    normalized = raw_value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise SettingsError(f"{name} must be a boolean value")
