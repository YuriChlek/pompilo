from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from bot_platform_service.trading_bots.spot_grid.domain import SpotGridConfig


@dataclass(frozen=True, slots=True)
class DecimalBounds:
    min: Decimal
    max: Decimal


@dataclass(frozen=True, slots=True)
class IntegerBounds:
    min: int
    max: int


@dataclass(frozen=True, slots=True)
class SpotGridConfigDefaults:
    symbols: tuple[str, ...]
    primary_timeframe: str
    supporting_timeframes: tuple[str, ...]
    max_position_fraction: Decimal
    max_grid_levels: int
    emit_diagnostics: bool
    min_price_distance_fraction: Decimal
    high_volatility_pause_threshold: Decimal
    volatility_cooldown_runs: int


SUPPORTED_TIMEFRAMES = ("1h", "4h")
MAX_POSITION_FRACTION_BOUNDS = DecimalBounds(min=Decimal("0.01"), max=Decimal("1.00"))
MAX_GRID_LEVELS_BOUNDS = IntegerBounds(min=1, max=50)
MIN_PRICE_DISTANCE_FRACTION_BOUNDS = DecimalBounds(min=Decimal("0"), max=Decimal("0.10"))
HIGH_VOLATILITY_PAUSE_THRESHOLD_BOUNDS = DecimalBounds(min=Decimal("0.001"), max=Decimal("1.00"))
VOLATILITY_COOLDOWN_RUNS_BOUNDS = IntegerBounds(min=0, max=20)
DEFAULT_SPOT_GRID_CONFIG = SpotGridConfigDefaults(
    symbols=("ETHUSDT",),
    primary_timeframe="1h",
    supporting_timeframes=("4h",),
    max_position_fraction=Decimal("0.10"),
    max_grid_levels=6,
    emit_diagnostics=True,
    min_price_distance_fraction=Decimal("0.001"),
    high_volatility_pause_threshold=Decimal("0.05"),
    volatility_cooldown_runs=2,
)


def parse_spot_grid_config(
    payload: Mapping[str, object],
    *,
    fallback_symbols: tuple[str, ...] = DEFAULT_SPOT_GRID_CONFIG.symbols,
    fallback_timeframes: tuple[str, ...] = (
        DEFAULT_SPOT_GRID_CONFIG.primary_timeframe,
        *DEFAULT_SPOT_GRID_CONFIG.supporting_timeframes,
    ),
) -> SpotGridConfig:
    """Resolve persisted instance overrides into a pure Spot Grid config."""

    symbols = _string_tuple(payload.get("symbols"), fallback=fallback_symbols or DEFAULT_SPOT_GRID_CONFIG.symbols)
    primary_timeframe = str(
        payload.get("primary_timeframe")
        or (fallback_timeframes[0] if fallback_timeframes else DEFAULT_SPOT_GRID_CONFIG.primary_timeframe)
    )
    supporting_timeframes = _string_tuple(
        payload.get("supporting_timeframes"),
        fallback=tuple(timeframe for timeframe in fallback_timeframes if timeframe != primary_timeframe),
    )
    return SpotGridConfig(
        symbols=tuple(symbol.upper() for symbol in symbols),
        primary_timeframe=primary_timeframe,
        supporting_timeframes=supporting_timeframes,
        max_position_fraction=_decimal_config_value(
            payload.get("max_position_fraction", DEFAULT_SPOT_GRID_CONFIG.max_position_fraction),
            field_name="max_position_fraction",
        ),
        max_grid_levels=int(payload.get("max_grid_levels", DEFAULT_SPOT_GRID_CONFIG.max_grid_levels)),
        emit_diagnostics=bool(payload.get("emit_diagnostics", DEFAULT_SPOT_GRID_CONFIG.emit_diagnostics)),
        min_price_distance_fraction=_decimal_config_value(
            payload.get("min_price_distance_fraction", DEFAULT_SPOT_GRID_CONFIG.min_price_distance_fraction),
            field_name="min_price_distance_fraction",
        ),
        high_volatility_pause_threshold=_decimal_config_value(
            payload.get("high_volatility_pause_threshold", DEFAULT_SPOT_GRID_CONFIG.high_volatility_pause_threshold),
            field_name="high_volatility_pause_threshold",
        ),
        volatility_cooldown_runs=int(
            payload.get("volatility_cooldown_runs", DEFAULT_SPOT_GRID_CONFIG.volatility_cooldown_runs)
        ),
    )


def _string_tuple(value: object, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple | list):
        parsed = tuple(str(item) for item in value if str(item).strip())
        if parsed:
            return parsed
    return fallback


def _decimal_config_value(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, float):
        raise TypeError(f"{field_name} must not be float")
    return Decimal(str(value))
