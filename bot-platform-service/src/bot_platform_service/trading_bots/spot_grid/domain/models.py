from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class GridLevelSide(StrEnum):
    """Side of a planned grid level."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class SpotGridCandle:
    """Pure candle model used by platform-native Spot Grid planning."""

    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class SpotGridConfig:
    """Validated planning settings for one Spot Grid run."""

    symbols: tuple[str, ...]
    primary_timeframe: str
    supporting_timeframes: tuple[str, ...]
    max_position_fraction: Decimal
    max_grid_levels: int
    emit_diagnostics: bool = True


@dataclass(frozen=True, slots=True)
class GridLevel:
    """One planned grid level before conversion to platform signals."""

    side: GridLevelSide
    price: Decimal
    level_index: int
    reason: str


@dataclass(frozen=True, slots=True)
class SpotGridPlan:
    """Pure Spot Grid planning result."""

    symbol: str
    timeframe: str
    reference_price: Decimal
    range_low: Decimal
    range_high: Decimal
    levels: tuple[GridLevel, ...]
    diagnostics: dict[str, object]
