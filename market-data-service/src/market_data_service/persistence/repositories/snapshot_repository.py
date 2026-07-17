from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.snapshot_hash import build_snapshot_membership, calculate_snapshot_data_hash
from market_data_service.domain.snapshot_models import MarketSnapshot, SnapshotCreationResult
from market_data_service.persistence.tables import market_candles, market_snapshot_candles, market_snapshots


class SnapshotRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def create_snapshot_if_changed(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        candles: Sequence[CanonicalCandle],
        batch_id: str,
        completeness_status: CandleRangeStatus,
    ) -> SnapshotCreationResult:
        if not candles:
            raise ValueError("Cannot create a snapshot without candles")

        ordered_candles = tuple(sorted(candles, key=lambda candle: candle.open_time))
        lookback_start_time = ordered_candles[0].open_time
        lookback_end_time = ordered_candles[-1].close_time
        data_hash = calculate_snapshot_data_hash(ordered_candles)

        existing_snapshot = await self._find_existing_snapshot(
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            lookback_start_time=lookback_start_time,
            lookback_end_time=lookback_end_time,
            data_hash=data_hash,
            completeness_status=completeness_status,
        )
        if existing_snapshot is not None:
            return SnapshotCreationResult(snapshot=existing_snapshot, created=False)

        snapshot_id = str(uuid4())
        snapshot_version = await self._next_snapshot_version(
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            lookback_start_time=lookback_start_time,
            lookback_end_time=lookback_end_time,
        )
        snapshot_insert = (
            market_snapshots.insert()
            .values(
                id=snapshot_id,
                source=source.value,
                canonical_symbol=canonical_symbol,
                timeframe=timeframe,
                last_closed_candle_time=ordered_candles[-1].close_time,
                lookback_start_time=lookback_start_time,
                lookback_end_time=lookback_end_time,
                candle_count=len(ordered_candles),
                data_hash=data_hash,
                batch_id=batch_id,
                completeness_status=completeness_status.value,
                snapshot_version=snapshot_version,
            )
            .returning(*market_snapshots.c)
        )
        snapshot_row = (await self.connection.execute(snapshot_insert)).mappings().one()
        membership_rows = [
            {
                "snapshot_id": membership.snapshot_id,
                "candle_id": membership.candle_id,
                "ordinal": membership.ordinal,
                "candle_hash_at_snapshot": membership.candle_hash_at_snapshot,
            }
            for membership in build_snapshot_membership(snapshot_id, ordered_candles)
        ]
        await self.connection.execute(market_snapshot_candles.insert().values(membership_rows))
        return SnapshotCreationResult(snapshot=_row_to_snapshot(snapshot_row), created=True)

    async def get_snapshot(self, snapshot_id: str) -> MarketSnapshot | None:
        query = select(market_snapshots).where(market_snapshots.c.id == snapshot_id)
        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            return None
        return _row_to_snapshot(row)

    async def get_latest_complete_snapshot(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ) -> MarketSnapshot | None:
        query = (
            select(market_snapshots)
            .where(
                market_snapshots.c.source == source.value,
                market_snapshots.c.canonical_symbol == canonical_symbol,
                market_snapshots.c.timeframe == timeframe,
                market_snapshots.c.completeness_status == CandleRangeStatus.COMPLETE.value,
            )
            .order_by(
                market_snapshots.c.last_closed_candle_time.desc(),
                market_snapshots.c.snapshot_version.desc(),
                market_snapshots.c.created_at.desc(),
            )
            .limit(1)
        )
        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            return None
        return _row_to_snapshot(row)

    async def read_snapshot_candles(self, snapshot_id: str) -> list[CanonicalCandle]:
        snapshot = await self.get_snapshot(snapshot_id)
        if snapshot is None:
            return []

        query = (
            select(market_candles)
            .select_from(
                market_snapshot_candles.join(
                    market_candles,
                    market_snapshot_candles.c.candle_id == market_candles.c.candle_id,
                )
            )
            .where(
                market_snapshot_candles.c.snapshot_id == snapshot_id,
                market_candles.c.source == snapshot.source.value,
                market_candles.c.canonical_symbol == snapshot.canonical_symbol,
                market_candles.c.timeframe == snapshot.timeframe,
            )
            .order_by(market_snapshot_candles.c.ordinal)
        )
        rows = (await self.connection.execute(query)).mappings().all()
        return [_row_to_candle(row) for row in rows]

    async def _find_existing_snapshot(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        lookback_start_time: datetime,
        lookback_end_time: datetime,
        data_hash: str,
        completeness_status: CandleRangeStatus,
    ) -> MarketSnapshot | None:
        query = select(market_snapshots).where(
            market_snapshots.c.source == source.value,
            market_snapshots.c.canonical_symbol == canonical_symbol,
            market_snapshots.c.timeframe == timeframe,
            market_snapshots.c.lookback_start_time == lookback_start_time,
            market_snapshots.c.lookback_end_time == lookback_end_time,
            market_snapshots.c.data_hash == data_hash,
            market_snapshots.c.completeness_status == completeness_status.value,
        )
        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            return None
        return _row_to_snapshot(row)

    async def _next_snapshot_version(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        lookback_start_time: datetime,
        lookback_end_time: datetime,
    ) -> int:
        query = (
            select(func.max(market_snapshots.c.snapshot_version))
            .where(
                market_snapshots.c.source == source.value,
                market_snapshots.c.canonical_symbol == canonical_symbol,
                market_snapshots.c.timeframe == timeframe,
                market_snapshots.c.lookback_start_time == lookback_start_time,
                market_snapshots.c.lookback_end_time == lookback_end_time,
            )
        )
        latest_version = (await self.connection.execute(query)).scalar_one_or_none()
        return int(latest_version or 0) + 1


def _row_to_snapshot(row) -> MarketSnapshot:
    return MarketSnapshot(
        id=row["id"],
        source=MarketDataSource(row["source"]),
        canonical_symbol=row["canonical_symbol"],
        timeframe=row["timeframe"],
        last_closed_candle_time=row["last_closed_candle_time"],
        lookback_start_time=row["lookback_start_time"],
        lookback_end_time=row["lookback_end_time"],
        candle_count=row["candle_count"],
        data_hash=row["data_hash"],
        batch_id=row["batch_id"],
        completeness_status=CandleRangeStatus(row["completeness_status"]),
        snapshot_version=row["snapshot_version"],
        created_at=row["created_at"],
    )


def _row_to_candle(row) -> CanonicalCandle:
    return CanonicalCandle(
        candle_id=row["candle_id"],
        source=MarketDataSource(row["source"]),
        canonical_symbol=row["canonical_symbol"],
        provider_symbol=row["provider_symbol"],
        timeframe=row["timeframe"],
        open_time=row["open_time"],
        close_time=row["close_time"],
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        volume=row["volume"],
        quote_volume=row["quote_volume"],
        taker_buy_base_volume=row["taker_buy_base_volume"],
        taker_buy_quote_volume=row["taker_buy_quote_volume"],
        taker_sell_base_volume=row["taker_sell_base_volume"],
        taker_sell_quote_volume=row["taker_sell_quote_volume"],
        trades_count=row["trades_count"],
        is_closed=row["is_closed"],
        provider_payload_hash=row["provider_payload_hash"],
    )
