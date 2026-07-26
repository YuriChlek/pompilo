from __future__ import annotations

from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository
from market_data_service.persistence.repositories.sync_job_repository import SyncJobRepository


class BackfillRequestRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection
        self.sync_job_repository = SyncJobRepository(connection)
        self.outbox_repository = OutboxRepository(connection)

    async def request_backfill(self, event: MarketDataBackfillRequested) -> bool:
        async with _transaction_boundary(self.connection):
            created = await self.sync_job_repository.enqueue_backfill_job(event)
            if created:
                await self.outbox_repository.create_pending_market_data_backfill_requested(event)
            return created


@asynccontextmanager
async def _transaction_boundary(connection: AsyncConnection):
    if connection.in_transaction():
        yield
    else:
        async with connection.begin():
            yield
