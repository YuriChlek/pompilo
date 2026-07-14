from __future__ import annotations

from enum import StrEnum


class MarketSymbolStatus(StrEnum):
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    DELISTED = "DELISTED"


class ProviderSymbolStatus(StrEnum):
    TRADING = "TRADING"
    HALTED = "HALTED"
    DELISTED = "DELISTED"


class MarketDataSource(StrEnum):
    BINANCE_SPOT = "BINANCE_SPOT"


class CandleRangeStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    GAP_DETECTED = "GAP_DETECTED"
    STALE = "STALE"


class MarketDataBatchStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INCOMPLETE = "INCOMPLETE"


class OutboxStatus(StrEnum):
    NOT_CREATED = "NOT_CREATED"
    PENDING = "PENDING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class SyncJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class SyncJobKind(StrEnum):
    FRESH = "FRESH"
    BACKFILL = "BACKFILL"
