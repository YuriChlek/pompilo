from __future__ import annotations

from datetime import datetime

from sqlalchemy import and_, delete, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.enums import OutboxStatus
from market_data_service.domain.events.candle_batch_ready import CandleBatchReady
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.domain.events.market_data_candles_collected import MarketDataCandlesCollectedEvent
from market_data_service.domain.outbox_models import OutboxEvent
from market_data_service.persistence.tables import outbox_events


class OutboxRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def create_pending_market_data_candles_collected(self, event: MarketDataCandlesCollectedEvent) -> bool:
        statement = (
            insert(outbox_events)
            .values(
                id=event.event_id,
                event_type=event.event_type,
                aggregate_type="market_snapshot",
                aggregate_id=event.snapshot_id,
                payload_json=event.payload_json(),
                idempotency_key=event.idempotency_key,
                status=OutboxStatus.PENDING.value,
                attempts=0,
            )
            .on_conflict_do_nothing(index_elements=["event_type", "idempotency_key"])
        )
        result = await self.connection.execute(statement)
        return int(result.rowcount or 0) > 0

    async def create_pending_candle_batch_ready(self, event: CandleBatchReady) -> bool:
        statement = (
            insert(outbox_events)
            .values(
                id=event.event_id,
                event_type=event.event_type,
                aggregate_type="market_snapshot",
                aggregate_id=event.snapshot_id,
                payload_json=event.payload_json(),
                idempotency_key=event.idempotency_key,
                status=OutboxStatus.PENDING.value,
                attempts=0,
            )
            .on_conflict_do_nothing(index_elements=["event_type", "idempotency_key"])
        )
        result = await self.connection.execute(statement)
        return int(result.rowcount or 0) > 0

    async def create_pending_market_data_backfill_requested(self, event: MarketDataBackfillRequested) -> bool:
        statement = (
            insert(outbox_events)
            .values(
                id=event.event_id,
                event_type=event.event_type,
                aggregate_type="sync_job",
                aggregate_id=event.idempotency_key,
                payload_json=event.payload_json(),
                idempotency_key=event.idempotency_key,
                status=OutboxStatus.PENDING.value,
                attempts=0,
            )
            .on_conflict_do_nothing(index_elements=["event_type", "idempotency_key"])
        )
        result = await self.connection.execute(statement)
        return int(result.rowcount or 0) > 0

    async def fetch_publishable_events(self, *, limit: int, now: datetime) -> list[OutboxEvent]:
        query = (
            select(outbox_events)
            .where(
                outbox_events.c.status == OutboxStatus.PENDING.value,
                or_(
                    outbox_events.c.next_attempt_at.is_(None),
                    outbox_events.c.next_attempt_at <= now,
                ),
            )
            .order_by(outbox_events.c.created_at, outbox_events.c.id)
            .limit(limit)
        )
        rows = (await self.connection.execute(query)).mappings().all()
        return [_row_to_outbox_event(row) for row in rows]

    async def list_replay_events(self, *, from_id: str | None, to_id: str | None) -> tuple[OutboxEvent, ...]:
        conditions = []
        if from_id:
            conditions.append(outbox_events.c.id >= from_id)
        if to_id:
            conditions.append(outbox_events.c.id <= to_id)
        query = select(outbox_events).order_by(outbox_events.c.created_at, outbox_events.c.id)
        if conditions:
            query = query.where(and_(*conditions))
        rows = (await self.connection.execute(query)).mappings().all()
        return tuple(_row_to_outbox_event(row) for row in rows)

    async def mark_published(self, *, event_id: str, published_at: datetime) -> None:
        statement = (
            update(outbox_events)
            .where(outbox_events.c.id == event_id)
            .values(
                status=OutboxStatus.PUBLISHED.value,
                published_at=published_at,
            )
        )
        await self.connection.execute(statement)

    async def mark_retry(self, *, event_id: str, attempts: int, next_attempt_at: datetime) -> None:
        statement = (
            update(outbox_events)
            .where(outbox_events.c.id == event_id)
            .values(
                status=OutboxStatus.PENDING.value,
                attempts=attempts,
                next_attempt_at=next_attempt_at,
            )
        )
        await self.connection.execute(statement)

    async def mark_failed(self, *, event_id: str, attempts: int) -> None:
        statement = (
            update(outbox_events)
            .where(outbox_events.c.id == event_id)
            .values(
                status=OutboxStatus.FAILED.value,
                attempts=attempts,
            )
        )
        await self.connection.execute(statement)

    async def delete_old_published_events(self, *, cutoff: datetime, batch_size: int = 1000) -> int:
        total_deleted = 0
        while True:
            select_stmt = (
                select(outbox_events.c.id)
                .where(
                    outbox_events.c.status == OutboxStatus.PUBLISHED.value,
                    outbox_events.c.published_at < cutoff,
                )
                .limit(batch_size)
            )
            result = await self.connection.execute(select_stmt)
            ids = [row[0] for row in result.all()]

            if not ids:
                break

            delete_stmt = delete(outbox_events).where(outbox_events.c.id.in_(ids))
            await self.connection.execute(delete_stmt)
            total_deleted += len(ids)

            if len(ids) < batch_size:
                break

        return total_deleted



def _row_to_outbox_event(row) -> OutboxEvent:
    return OutboxEvent(
        id=row["id"],
        event_type=row["event_type"],
        aggregate_type=row["aggregate_type"],
        aggregate_id=row["aggregate_id"],
        payload=row["payload_json"],
        idempotency_key=row["idempotency_key"],
        status=OutboxStatus(row["status"]),
        attempts=row["attempts"],
        next_attempt_at=row["next_attempt_at"],
        created_at=row["created_at"],
        published_at=row["published_at"],
    )
