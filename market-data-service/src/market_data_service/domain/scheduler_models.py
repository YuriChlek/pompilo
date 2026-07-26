from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_data_service.domain.enums import MarketDataSource, SyncJobKind

FRESH_SYNC_PRIORITY = 10
BACKFILL_SYNC_PRIORITY = 100


@dataclass(frozen=True, slots=True)
class MarketDataSyncJob:
    source: MarketDataSource
    provider_symbol: str
    timeframe: str
    expected_close_time: datetime
    scheduled_for: datetime
    idempotency_key: str
    job_kind: SyncJobKind = SyncJobKind.FRESH
    priority: int = FRESH_SYNC_PRIORITY
    requested_from: datetime | None = None
    requested_to: datetime | None = None


@dataclass(frozen=True, slots=True)
class SchedulerTickResult:
    created_count: int
    skipped_count: int
