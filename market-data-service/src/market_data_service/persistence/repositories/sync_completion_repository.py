from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import CandleRangeStatus, MarketDataBatchStatus, MarketDataSource
from market_data_service.domain.events.candle_batch_ready import CandleBatchReady
from market_data_service.domain.snapshot_models import SnapshotCreationResult
from market_data_service.persistence.repositories.batch_repository import BatchRepository
from market_data_service.persistence.repositories.candle_repository import CandleRepository
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository
from market_data_service.persistence.repositories.snapshot_repository import SnapshotRepository


class SyncCompletionRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection
        self.candle_repository = CandleRepository(connection)
        self.batch_repository = BatchRepository(connection)
        self.snapshot_repository = SnapshotRepository(connection)
        self.outbox_repository = OutboxRepository(connection)

    async def complete_sync(
        self,
        *,
        batch_id: str,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        candles: list[CanonicalCandle],
        batch_status: MarketDataBatchStatus,
        rows_fetched: int,
        dry_run: bool,
        gap_count: int,
        first_open_time: datetime | None,
        last_close_time: datetime | None,
    ) -> tuple[int, int, SnapshotCreationResult | None]:
        async with _transaction_boundary(self.connection):
            inserted_count = 0 if dry_run else await self.candle_repository.insert_closed_candles(candles)
            skipped_duplicate_count = 0 if dry_run else len(candles) - inserted_count

            await self.batch_repository.complete_batch(
                batch_id=batch_id,
                status=batch_status,
                rows_fetched=rows_fetched,
                rows_inserted=inserted_count,
                rows_skipped_duplicate=skipped_duplicate_count,
                rows_hash_mismatch=0,
                gap_count=gap_count,
                first_open_time=first_open_time,
                last_close_time=last_close_time,
            )

            snapshot_result = None
            if batch_status == MarketDataBatchStatus.COMPLETE and not dry_run:
                snapshot_result = await self.snapshot_repository.create_snapshot_if_changed(
                    source=source,
                    canonical_symbol=canonical_symbol,
                    timeframe=timeframe,
                    candles=candles,
                    batch_id=batch_id,
                    completeness_status=CandleRangeStatus.COMPLETE,
                )
                await self.outbox_repository.create_pending_candle_batch_ready(
                    CandleBatchReady.from_snapshot(snapshot_result.snapshot)
                )

            return inserted_count, skipped_duplicate_count, snapshot_result


@asynccontextmanager
async def _transaction_boundary(connection: AsyncConnection):
    if connection.in_transaction():
        yield
    else:
        async with connection.begin():
            yield
