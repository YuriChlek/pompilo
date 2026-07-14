from __future__ import annotations

from typing import Protocol

from market_data_service.domain.scheduler_models import MarketDataSyncJob


class SyncJobQueuePort(Protocol):
    async def enqueue_sync_job(self, job: MarketDataSyncJob) -> bool: ...
