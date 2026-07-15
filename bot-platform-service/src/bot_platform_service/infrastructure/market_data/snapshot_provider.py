from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncConnection

from bot_platform_service.domain import (
    BotCandle,
    BotMarketDataContext,
    BotMarketSnapshot,
    UnsupportedTimeframeError,
    normalize_timeframe,
)
from bot_platform_service.infrastructure.market_data.errors import (
    SnapshotNotReadyError,
    SnapshotStaleError,
    TimeframeUnsupportedError,
)
from bot_platform_service.infrastructure.market_data.tables import market_candles, market_snapshot_candles, market_snapshots

COMPLETE_STATUS = "COMPLETE"


class MarketDataServiceSnapshotProvider:
    """Read-through provider over Market Data Service immutable snapshots."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def get_latest_complete_snapshot(
        self,
        *,
        source: str,
        canonical_symbol: str,
        timeframe: str,
        min_snapshot_version: int | None = None,
    ) -> BotMarketSnapshot:
        """Return the latest complete snapshot and exact candle membership."""
        canonical_timeframe = _normalize_timeframe_or_raise(timeframe)
        query = (
            select(market_snapshots)
            .where(
                market_snapshots.c.source == source,
                market_snapshots.c.canonical_symbol == canonical_symbol,
                market_snapshots.c.timeframe == canonical_timeframe,
                market_snapshots.c.completeness_status == COMPLETE_STATUS,
            )
            .order_by(
                desc(market_snapshots.c.last_closed_candle_time),
                desc(market_snapshots.c.snapshot_version),
            )
            .limit(1)
        )
        if min_snapshot_version is not None:
            query = query.where(market_snapshots.c.snapshot_version >= min_snapshot_version)

        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            raise SnapshotNotReadyError(
                f"No complete snapshot for source={source} symbol={canonical_symbol} timeframe={canonical_timeframe}"
            )
        return await self._snapshot_from_row(row)

    async def build_context(
        self,
        *,
        source: str,
        canonical_symbol: str,
        primary_timeframe: str,
        supporting_timeframes: tuple[str, ...],
    ) -> BotMarketDataContext:
        """Return primary and supporting snapshots for one symbol."""
        primary_snapshot = await self.get_latest_complete_snapshot(
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=primary_timeframe,
        )
        supporting_snapshots = tuple(
            [
                await self.get_latest_complete_snapshot(
                    source=source,
                    canonical_symbol=canonical_symbol,
                    timeframe=timeframe,
                )
                for timeframe in supporting_timeframes
            ]
        )
        return BotMarketDataContext(primary_snapshot=primary_snapshot, supporting_snapshots=supporting_snapshots)

    async def _snapshot_from_row(self, row: Mapping[str, object]) -> BotMarketSnapshot:
        snapshot_id = str(row["id"])
        candles, provider_symbol = await self._read_snapshot_candles(
            snapshot_id=snapshot_id,
            source=str(row["source"]),
            canonical_symbol=str(row["canonical_symbol"]),
            timeframe=str(row["timeframe"]),
        )
        if len(candles) != int(row["candle_count"]):
            raise SnapshotStaleError(
                f"Snapshot {snapshot_id} expected {row['candle_count']} candles but read {len(candles)}"
            )
        return BotMarketSnapshot(
            snapshot_id=snapshot_id,
            source=str(row["source"]),
            canonical_symbol=str(row["canonical_symbol"]),
            provider_symbol=provider_symbol,
            timeframe=str(row["timeframe"]),
            last_closed_candle_time=row["last_closed_candle_time"],
            lookback_start_time=row["lookback_start_time"],
            lookback_end_time=row["lookback_end_time"],
            completeness_status=str(row["completeness_status"]),
            snapshot_version=int(row["snapshot_version"]),
            data_hash=str(row["data_hash"]),
            candles=tuple(candles),
        )

    async def _read_snapshot_candles(
        self,
        *,
        snapshot_id: str,
        source: str,
        canonical_symbol: str,
        timeframe: str,
    ) -> tuple[list[BotCandle], str]:
        query = (
            select(
                market_snapshot_candles.c.candle_hash_at_snapshot,
                market_candles.c.source,
                market_candles.c.canonical_symbol,
                market_candles.c.provider_symbol,
                market_candles.c.timeframe,
                market_candles.c.open_time,
                market_candles.c.close_time,
                market_candles.c.open,
                market_candles.c.high,
                market_candles.c.low,
                market_candles.c.close,
                market_candles.c.volume,
                market_candles.c.provider_payload_hash,
            )
            .select_from(
                market_snapshot_candles.join(
                    market_candles,
                    market_snapshot_candles.c.candle_id == market_candles.c.candle_id,
                )
            )
            .where(
                market_snapshot_candles.c.snapshot_id == snapshot_id,
                market_candles.c.source == source,
                market_candles.c.canonical_symbol == canonical_symbol,
                market_candles.c.timeframe == timeframe,
            )
            .order_by(market_snapshot_candles.c.ordinal)
        )
        rows = (await self.connection.execute(query)).mappings().all()
        candles: list[BotCandle] = []
        provider_symbol = canonical_symbol
        for row in rows:
            if row["candle_hash_at_snapshot"] != row["provider_payload_hash"]:
                raise SnapshotStaleError(f"Snapshot {snapshot_id} candle hash mismatch")
            provider_symbol = str(row["provider_symbol"])
            candles.append(_candle_from_row(row))
        return candles, provider_symbol


def _normalize_timeframe_or_raise(timeframe: str) -> str:
    try:
        return normalize_timeframe(timeframe)
    except UnsupportedTimeframeError as exc:
        raise TimeframeUnsupportedError(str(exc)) from exc


def _candle_from_row(row: Mapping[str, object]) -> BotCandle:
    return BotCandle(
        source=str(row["source"]),
        canonical_symbol=str(row["canonical_symbol"]),
        timeframe=str(row["timeframe"]),
        open_time=row["open_time"],
        close_time=row["close_time"],
        open=Decimal(str(row["open"])),
        high=Decimal(str(row["high"])),
        low=Decimal(str(row["low"])),
        close=Decimal(str(row["close"])),
        volume=Decimal(str(row["volume"])),
    )
