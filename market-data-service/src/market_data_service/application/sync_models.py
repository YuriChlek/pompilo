from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_data_service.domain.enums import CandleRangeStatus, MarketDataBatchStatus, MarketDataSource


@dataclass(frozen=True, slots=True)
class SyncClosedCandlesCommand:
    source: MarketDataSource
    provider_symbol: str
    timeframe: str
    from_time: datetime
    to_time: datetime
    dry_run: bool = False
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class SyncClosedCandlesResult:
    batch_id: str
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    timeframe: str
    fetched_count: int
    inserted_count: int
    skipped_duplicate_count: int
    range_status: CandleRangeStatus
    batch_status: MarketDataBatchStatus
    gap_count: int
    dry_run: bool
    snapshot_id: str | None = None
    snapshot_created: bool = False
