from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    id: str
    source: MarketDataSource
    canonical_symbol: str
    timeframe: str
    last_closed_candle_time: datetime
    lookback_start_time: datetime
    lookback_end_time: datetime
    candle_count: int
    data_hash: str
    batch_id: str
    completeness_status: CandleRangeStatus
    snapshot_version: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MarketSnapshotCandle:
    snapshot_id: str
    candle_id: str
    ordinal: int
    candle_hash_at_snapshot: str


@dataclass(frozen=True, slots=True)
class SnapshotCreationResult:
    snapshot: MarketSnapshot
    created: bool
