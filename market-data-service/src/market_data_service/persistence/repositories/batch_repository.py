from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.batch_models import MarketDataBatch
from market_data_service.domain.enums import MarketDataBatchStatus, MarketDataSource, OutboxStatus
from market_data_service.persistence.tables import market_data_batches


class BatchRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def create_running_batch(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        requested_from: datetime,
        requested_to: datetime,
        expected_close_time: datetime,
    ) -> MarketDataBatch:
        batch_id = str(uuid4())
        statement = (
            market_data_batches.insert()
            .values(
                batch_id=batch_id,
                source=source.value,
                canonical_symbol=canonical_symbol,
                timeframe=timeframe,
                requested_from=requested_from,
                requested_to=requested_to,
                expected_close_time=expected_close_time,
                status=MarketDataBatchStatus.RUNNING.value,
                outbox_status=OutboxStatus.NOT_CREATED.value,
                rows_fetched=0,
                rows_inserted=0,
                rows_skipped_duplicate=0,
                rows_hash_mismatch=0,
                gap_count=0,
            )
            .returning(*market_data_batches.c)
        )
        result = await self.connection.execute(statement)
        return _row_to_batch(result.mappings().one())

    async def complete_batch(
        self,
        *,
        batch_id: str,
        status: MarketDataBatchStatus,
        rows_fetched: int,
        rows_inserted: int,
        rows_skipped_duplicate: int,
        rows_hash_mismatch: int,
        gap_count: int,
        first_open_time: datetime | None,
        last_close_time: datetime | None,
    ) -> None:
        statement = (
            update(market_data_batches)
            .where(market_data_batches.c.batch_id == batch_id)
            .values(
                status=status.value,
                rows_fetched=rows_fetched,
                rows_inserted=rows_inserted,
                rows_skipped_duplicate=rows_skipped_duplicate,
                rows_hash_mismatch=rows_hash_mismatch,
                gap_count=gap_count,
                first_open_time=first_open_time,
                last_close_time=last_close_time,
                completed_at=datetime.now(UTC),
            )
        )
        await self.connection.execute(statement)

    async def fail_batch(
        self,
        *,
        batch_id: str,
        error_code: str,
        error_message_redacted: str,
    ) -> None:
        statement = (
            update(market_data_batches)
            .where(market_data_batches.c.batch_id == batch_id)
            .values(
                status=MarketDataBatchStatus.FAILED.value,
                error_code=error_code,
                error_message_redacted=error_message_redacted,
                completed_at=datetime.now(UTC),
            )
        )
        await self.connection.execute(statement)


def _row_to_batch(row) -> MarketDataBatch:
    return MarketDataBatch(
        batch_id=row["batch_id"],
        source=MarketDataSource(row["source"]),
        canonical_symbol=row["canonical_symbol"],
        timeframe=row["timeframe"],
        requested_from=row["requested_from"],
        requested_to=row["requested_to"],
        expected_close_time=row["expected_close_time"],
        status=MarketDataBatchStatus(row["status"]),
        outbox_status=OutboxStatus(row["outbox_status"]),
        rows_fetched=row["rows_fetched"],
        rows_inserted=row["rows_inserted"],
        rows_skipped_duplicate=row["rows_skipped_duplicate"],
        rows_hash_mismatch=row["rows_hash_mismatch"],
        gap_count=row["gap_count"],
        first_open_time=row["first_open_time"],
        last_close_time=row["last_close_time"],
        error_code=row["error_code"],
        error_message_redacted=row["error_message_redacted"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )
