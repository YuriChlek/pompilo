from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from bot_platform_service.domain import BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.domain import (
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlan,
    SpotGridPlanner,
)


@dataclass(frozen=True, slots=True)
class SpotGridCycleResult:
    """One platform-native Spot Grid planning result."""

    plan: SpotGridPlan
    diagnostics: dict[str, object]


class SpotGridTradingCycleService:
    """Application service for one side-effect-free Spot Grid planning cycle."""

    def __init__(self, planner: SpotGridPlanner | None = None) -> None:
        self.planner = planner or SpotGridPlanner()

    def run_once(
        self,
        *,
        snapshot: BotMarketSnapshot,
        config: SpotGridConfig,
    ) -> SpotGridCycleResult:
        """Plan one Spot Grid cycle from a platform market snapshot."""
        candles = tuple(
            SpotGridCandle(
                timestamp=candle.close_time.isoformat(),
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            for candle in snapshot.candles
        )
        plan = self.planner.plan(
            symbol=snapshot.canonical_symbol,
            timeframe=snapshot.timeframe,
            candles=candles,
            config=config,
        )
        return SpotGridCycleResult(
            plan=plan,
            diagnostics={
                **plan.diagnostics,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_version": snapshot.snapshot_version,
                "data_hash": snapshot.data_hash,
            },
        )


def parse_spot_grid_config(
    payload: Mapping[str, object],
    *,
    fallback_symbols: tuple[str, ...],
    fallback_timeframes: tuple[str, ...],
) -> SpotGridConfig:
    """Parse persisted module config into SpotGridConfig without side effects."""
    symbols = _string_tuple(payload.get("symbols"), fallback=fallback_symbols)
    primary_timeframe = str(payload.get("primary_timeframe") or (fallback_timeframes[0] if fallback_timeframes else "1h"))
    supporting_timeframes = _string_tuple(payload.get("supporting_timeframes"), fallback=tuple(timeframe for timeframe in fallback_timeframes if timeframe != primary_timeframe))
    return SpotGridConfig(
        symbols=tuple(symbol.upper() for symbol in symbols),
        primary_timeframe=primary_timeframe,
        supporting_timeframes=supporting_timeframes,
        max_position_fraction=Decimal(str(payload.get("max_position_fraction", "0.10"))),
        max_grid_levels=int(payload.get("max_grid_levels", 6)),
        emit_diagnostics=bool(payload.get("emit_diagnostics", True)),
    )


def _string_tuple(value: object, *, fallback: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple | list):
        parsed = tuple(str(item) for item in value if str(item).strip())
        if parsed:
            return parsed
    return fallback
