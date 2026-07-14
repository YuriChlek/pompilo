from __future__ import annotations

from datetime import datetime
from typing import Protocol

from market_data_service.domain.batch_models import MarketDataBatch
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import MarketDataBatchStatus, MarketDataSource
from market_data_service.domain.snapshot_models import SnapshotCreationResult
from market_data_service.domain.symbol_registry_models import ProviderSymbol


class CandleProviderPort(Protocol):
    async def fetch_closed_candles(
        self,
        provider_symbol: ProviderSymbol,
        *,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[CanonicalCandle]: ...


class CandleWriterPort(Protocol):
    async def insert_closed_candles(self, candles: list[CanonicalCandle]) -> int: ...


class AdvisoryLockPort(Protocol):
    async def acquire_sync_lock(self, *, source: str, provider_symbol: str, timeframe: str) -> bool: ...


class BatchTrackerPort(Protocol):
    async def create_running_batch(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        requested_from: datetime,
        requested_to: datetime,
        expected_close_time: datetime,
    ) -> MarketDataBatch: ...

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
    ) -> None: ...

    async def fail_batch(
        self,
        *,
        batch_id: str,
        error_code: str,
        error_message_redacted: str,
    ) -> None: ...


class SyncCompletionPort(Protocol):
    async def complete_sync(
        self,
        *,
        batch_id: str,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        candles: list[CanonicalCandle],
        batch_status: MarketDataBatchStatus,
        rows_fetched: int,
        dry_run: bool,
        gap_count: int,
        first_open_time: datetime | None,
        last_close_time: datetime | None,
    ) -> tuple[int, int, SnapshotCreationResult | None]: ...
