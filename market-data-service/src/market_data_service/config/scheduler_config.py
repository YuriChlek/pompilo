from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.scheduler import SUPPORTED_SCHEDULE_TIMEFRAMES
from market_data_service.domain.symbol_normalization import normalize_symbol


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    source: MarketDataSource
    provider_symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    safety_delay_by_timeframe: dict[str, timedelta]
    jitter_seconds: int
    poll_interval_seconds: float


def get_scheduler_config() -> SchedulerConfig:
    timeframes = _parse_csv(os.getenv("MARKET_DATA_TIMEFRAMES", "1h,4h,1d"), lowercase=True)
    _validate_supported_timeframes(timeframes)
    jitter_seconds = _parse_non_negative_int("MARKET_DATA_SCHEDULER_JITTER_SECONDS", default="30")
    poll_interval_seconds = _parse_positive_float(
        "MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS",
        default=os.getenv("MARKET_DATA_SCHEDULER_POLL_INTERVAL_SECONDS", "30"),
    )

    return SchedulerConfig(
        source=MarketDataSource(os.getenv("MARKET_DATA_SOURCE", MarketDataSource.BINANCE_SPOT.value)),
        provider_symbols=tuple(
            normalize_symbol(symbol)
            for symbol in _parse_csv(os.getenv("MARKET_DATA_PROVIDER_SYMBOLS", "ETHUSDT"))
        ),
        timeframes=timeframes,
        safety_delay_by_timeframe={
            "1h": timedelta(seconds=_parse_non_negative_int("MARKET_DATA_1H_SAFETY_DELAY_SECONDS", default="30")),
            "4h": timedelta(seconds=_parse_non_negative_int("MARKET_DATA_4H_SAFETY_DELAY_SECONDS", default="45")),
            "1d": timedelta(seconds=_parse_non_negative_int("MARKET_DATA_1D_SAFETY_DELAY_SECONDS", default="90")),
        },
        jitter_seconds=jitter_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def _parse_csv(value: str, *, lowercase: bool = False) -> tuple[str, ...]:
    items = tuple(item.strip().lower() if lowercase else item.strip().upper() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("CSV environment value must contain at least one item")
    return items


def _validate_supported_timeframes(timeframes: tuple[str, ...]) -> None:
    unsupported = sorted(set(timeframes).difference(SUPPORTED_SCHEDULE_TIMEFRAMES))
    if unsupported:
        raise ValueError(f"Unsupported scheduler timeframe(s): {', '.join(unsupported)}")


def _parse_non_negative_int(name: str, *, default: str) -> int:
    value = int(os.getenv(name, default))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _parse_positive_float(name: str, *, default: str) -> float:
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
