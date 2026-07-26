from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_data_service.domain.enums import MarketDataBatchStatus, MarketDataSource, OutboxStatus


@dataclass(frozen=True, slots=True)
class MarketDataBatch:
    batch_id: str
    source: MarketDataSource
    canonical_symbol: str
    timeframe: str
    requested_from: datetime
    requested_to: datetime
    expected_close_time: datetime
    status: MarketDataBatchStatus
    outbox_status: OutboxStatus
    rows_fetched: int
    rows_inserted: int
    rows_skipped_duplicate: int
    rows_hash_mismatch: int
    gap_count: int
    first_open_time: datetime | None
    last_close_time: datetime | None
    error_code: str | None
    error_message_redacted: str | None
    started_at: datetime
    completed_at: datetime | None
