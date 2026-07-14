from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.enums import MarketDataSource, SyncJobKind, SyncJobStatus
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.domain.scheduler_models import MarketDataSyncJob
from market_data_service.persistence.tables import sync_jobs


class SyncJobRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def enqueue_sync_job(self, job: MarketDataSyncJob) -> bool:
        statement = (
            insert(sync_jobs)
            .values(
                idempotency_key=job.idempotency_key,
                source=job.source.value,
                provider_symbol=job.provider_symbol,
                timeframe=job.timeframe,
                job_kind=job.job_kind.value,
                priority=job.priority,
                requested_from=job.requested_from,
                requested_to=job.requested_to,
                expected_close_time=job.expected_close_time,
                scheduled_for=job.scheduled_for,
                status=SyncJobStatus.PENDING.value,
                attempts=0,
            )
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
        )
        result = await self.connection.execute(statement)
        return int(result.rowcount or 0) > 0

    async def enqueue_backfill_job(self, event: MarketDataBackfillRequested) -> bool:
        job = MarketDataSyncJob(
            source=event.source,
            provider_symbol=event.provider_symbol,
            timeframe=event.timeframe,
            requested_from=event.requested_from,
            requested_to=event.requested_to,
            expected_close_time=event.requested_to,
            scheduled_for=event.occurred_at,
            idempotency_key=f"{SyncJobKind.BACKFILL.value}|{event.idempotency_key}",
            job_kind=SyncJobKind.BACKFILL,
            priority=event.priority,
        )
        return await self.enqueue_sync_job(job)

    async def claim_next_pending_job(self, *, now) -> MarketDataSyncJob | None:
        query = (
            select(sync_jobs)
            .where(
                sync_jobs.c.status == SyncJobStatus.PENDING.value,
                sync_jobs.c.scheduled_for <= now,
            )
            .order_by(sync_jobs.c.priority, sync_jobs.c.scheduled_for, sync_jobs.c.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            return None
        statement = (
            update(sync_jobs)
            .where(sync_jobs.c.idempotency_key == row["idempotency_key"])
            .values(status=SyncJobStatus.RUNNING.value, attempts=sync_jobs.c.attempts + 1, started_at=now)
        )
        await self.connection.execute(statement)
        return _row_to_sync_job(row)


def _row_to_sync_job(row) -> MarketDataSyncJob:
    return MarketDataSyncJob(
        source=MarketDataSource(row["source"]),
        provider_symbol=row["provider_symbol"],
        timeframe=row["timeframe"],
        expected_close_time=row["expected_close_time"],
        scheduled_for=row["scheduled_for"],
        idempotency_key=row["idempotency_key"],
        job_kind=SyncJobKind(row["job_kind"]),
        priority=row["priority"],
        requested_from=row["requested_from"],
        requested_to=row["requested_to"],
    )
