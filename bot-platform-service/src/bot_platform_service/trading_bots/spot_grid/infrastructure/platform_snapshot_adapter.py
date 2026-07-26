from __future__ import annotations

from dataclasses import dataclass

from bot_platform_service.domain import BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.domain import IndicatorCandle, IndicatorInput

DEFAULT_INDICATOR_REQUIRED_HISTORY = 200


@dataclass(frozen=True, slots=True)
class PlatformSnapshotIndicatorAdapter:
    """Adapt platform market snapshots into Spot Grid indicator input."""

    required_history: int = DEFAULT_INDICATOR_REQUIRED_HISTORY

    def adapt(self, snapshot: BotMarketSnapshot) -> IndicatorInput:
        candles = tuple(
            IndicatorCandle(
                timestamp=candle.close_time.isoformat(),
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            for candle in sorted(snapshot.candles, key=lambda candle: candle.close_time)
        )
        return IndicatorInput(
            source=snapshot.source,
            symbol=snapshot.canonical_symbol,
            timeframe=snapshot.timeframe,
            snapshot_id=snapshot.snapshot_id,
            snapshot_version=snapshot.snapshot_version,
            data_hash=snapshot.data_hash,
            candles=candles,
            required_history=self.required_history,
        )


__all__ = [
    "DEFAULT_INDICATOR_REQUIRED_HISTORY",
    "PlatformSnapshotIndicatorAdapter",
]
