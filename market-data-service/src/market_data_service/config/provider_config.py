from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os


@dataclass(frozen=True, slots=True)
class BinanceSpotProviderConfig:
    rest_endpoint: str
    request_timeout_seconds: float
    max_limit: int
    safety_delay_by_timeframe: dict[str, timedelta]
    max_concurrent_requests: int = 8
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: float = 60.0


def get_binance_spot_provider_config() -> BinanceSpotProviderConfig:
    return BinanceSpotProviderConfig(
        rest_endpoint=os.getenv("BINANCE_REST_ENDPOINT", "https://api.binance.com").rstrip("/"),
        request_timeout_seconds=float(os.getenv("BINANCE_REQUEST_TIMEOUT_SECONDS", "30")),
        max_limit=int(os.getenv("BINANCE_KLINE_LIMIT", "1000")),
        safety_delay_by_timeframe={
            "1h": timedelta(seconds=int(os.getenv("MARKET_DATA_1H_SAFETY_DELAY_SECONDS", "30"))),
            "4h": timedelta(seconds=int(os.getenv("MARKET_DATA_4H_SAFETY_DELAY_SECONDS", "45"))),
            "1d": timedelta(seconds=int(os.getenv("MARKET_DATA_1D_SAFETY_DELAY_SECONDS", "90"))),
        },
        max_concurrent_requests=_parse_positive_int("BINANCE_MAX_CONCURRENT_REQUESTS", default="8"),
        circuit_breaker_failure_threshold=_parse_positive_int("BINANCE_CIRCUIT_BREAKER_FAILURE_THRESHOLD", default="5"),
        circuit_breaker_recovery_timeout_seconds=_parse_positive_float(
            "BINANCE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS",
            default="60",
        ),
    )


def _parse_positive_int(name: str, *, default: str) -> int:
    value = int(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _parse_positive_float(name: str, *, default: str) -> float:
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
