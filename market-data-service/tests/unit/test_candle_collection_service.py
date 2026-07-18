from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest
from typing import Mapping

from market_data_service.application.services.candle_collection_service import (
    CandleCollectionService,
    CollectionResult,
)
from market_data_service.application.services.multi_provider_symbol_resolver import ResolvedProvider
from market_data_service.domain.collection_models import CandleCollectionState
from market_data_service.application.sync_models import SyncClosedCandlesCommand, SyncClosedCandlesResult
from market_data_service.domain.enums import MarketDataSource, CandleRangeStatus, MarketDataBatchStatus, SyncJobKind
from market_data_service.domain.scheduler_models import MarketDataSyncJob
from market_data_service.domain.scheduler import due_closed_candle_time


class MockResolver:
    def __init__(self) -> None:
        self.resolved_symbols: dict[str, ResolvedProvider | None] = {}
        self.calls: list[str] = []

    async def resolve(self, requested_symbol: str) -> ResolvedProvider | None:
        self.calls.append(requested_symbol)
        return self.resolved_symbols.get(requested_symbol)


class MockSingleSymbolSyncService:
    def __init__(self) -> None:
        self.calls: list[SyncClosedCandlesCommand] = []
        self.fail_for_symbol: str | None = None

    async def sync_closed_candles(self, command: SyncClosedCandlesCommand) -> SyncClosedCandlesResult:
        self.calls.append(command)
        if self.fail_for_symbol and command.provider_symbol == self.fail_for_symbol:
            raise RuntimeError(f"Sync failed for {command.provider_symbol}")
        return SyncClosedCandlesResult(
            batch_id="batch-1",
            source=command.source,
            canonical_symbol=command.provider_symbol,
            provider_symbol=command.provider_symbol,
            timeframe=command.timeframe,
            fetched_count=10,
            inserted_count=10,
            skipped_duplicate_count=0,
            range_status=CandleRangeStatus.COMPLETE,
            batch_status=MarketDataBatchStatus.COMPLETE,
            gap_count=0,
            dry_run=False,
        )


class MockSyncJobQueue:
    def __init__(self) -> None:
        self.enqueued_jobs: list[MarketDataSyncJob] = []
        self.pending_jobs: list[MarketDataSyncJob] = []
        self.completed_keys: list[str] = []
        self.failed_keys: list[str] = []

    async def enqueue_sync_job(self, job: MarketDataSyncJob) -> bool:
        self.enqueued_jobs.append(job)
        self.pending_jobs.append(job)
        return True

    async def claim_next_pending_job(self, *, now: datetime) -> MarketDataSyncJob | None:
        if not self.pending_jobs:
            return None
        return self.pending_jobs.pop(0)

    async def mark_completed(self, *, idempotency_key: str, completed_at: datetime) -> None:
        self.completed_keys.append(idempotency_key)

    async def mark_failed(self, *, idempotency_key: str, completed_at: datetime) -> None:
        self.failed_keys.append(idempotency_key)


class MockOutboxPublishBatchResult:
    def __init__(self, published_count: int) -> None:
        self.published_count = published_count


class MockOutboxPublisher:
    def __init__(self) -> None:
        self.calls = 0
        self.published_count = 5

    async def publish_once(self) -> MockOutboxPublishBatchResult:
        self.calls += 1
        return MockOutboxPublishBatchResult(self.published_count)


class MockCollectionStateRepository:
    def __init__(self) -> None:
        self.states: dict[tuple[MarketDataSource, str, str], CandleCollectionState] = {}

    async def get(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ) -> CandleCollectionState | None:
        return self.states.get((source, canonical_symbol, timeframe))

    async def create_bootstrap_state(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        bootstrap_from: datetime,
        bootstrap_to: datetime,
    ) -> CandleCollectionState:
        state = CandleCollectionState(
            source=source,
            canonical_symbol=canonical_symbol,
            provider_symbol=provider_symbol,
            timeframe=timeframe,
            bootstrap_from=bootstrap_from,
            bootstrap_to=bootstrap_to,
            bootstrap_next_from=bootstrap_from,
            bootstrap_completed_at=None,
            last_successful_close_time=None,
        )
        self.states[(source, canonical_symbol, timeframe)] = state
        return state

    async def mark_bootstrap_progress(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        provider_symbol: str,
        bootstrap_next_from: datetime,
        completed_at: datetime | None,
        last_successful_close_time: datetime | None,
    ) -> None:
        current = self.states[(source, canonical_symbol, timeframe)]
        self.states[(source, canonical_symbol, timeframe)] = CandleCollectionState(
            source=source,
            canonical_symbol=canonical_symbol,
            provider_symbol=provider_symbol,
            timeframe=timeframe,
            bootstrap_from=current.bootstrap_from,
            bootstrap_to=current.bootstrap_to,
            bootstrap_next_from=bootstrap_next_from,
            bootstrap_completed_at=completed_at,
            last_successful_close_time=last_successful_close_time or current.last_successful_close_time,
        )

    async def mark_incremental_progress(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        provider_symbol: str,
        last_successful_close_time: datetime,
        updated_at: datetime,
    ) -> None:
        current = self.states[(source, canonical_symbol, timeframe)]
        self.states[(source, canonical_symbol, timeframe)] = CandleCollectionState(
            source=source,
            canonical_symbol=canonical_symbol,
            provider_symbol=provider_symbol,
            timeframe=timeframe,
            bootstrap_from=current.bootstrap_from,
            bootstrap_to=current.bootstrap_to,
            bootstrap_next_from=current.bootstrap_next_from,
            bootstrap_completed_at=current.bootstrap_completed_at,
            last_successful_close_time=last_successful_close_time,
        )


class MockCandleHistory:
    def __init__(self) -> None:
        # Key: (source, symbol, timeframe) -> latest_time
        self.history: dict[tuple[MarketDataSource, str, str], datetime | None] = {}

    async def get_latest_closed_candle_time(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ) -> datetime | None:
        return self.history.get((source, canonical_symbol, timeframe))


class CandleCollectionServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 18, 12, tzinfo=UTC)
        self.resolver = MockResolver()
        self.sync_service = MockSingleSymbolSyncService()
        self.queue = MockSyncJobQueue()
        self.history = MockCandleHistory()
        self.publisher = MockOutboxPublisher()
        self.collection_state = MockCollectionStateRepository()

        self.service = CandleCollectionService(
            resolver=self.resolver,
            single_symbol_sync=self.sync_service,
            sync_job_queue=self.queue,
            candle_history=self.history,
            collection_state=self.collection_state,
            provider_symbols=("BTCUSDT", "SOLUSDT", "UNKNOWN"),
            timeframes=("1h",),
            safety_delay_by_timeframe={"1h": timedelta(minutes=5)},
            jitter_seconds=0,
            bootstrap_lookback_years=2,
            bootstrap_max_chunks_per_tick=3,
            chunk_limit=100,  # 1 chunk is 100 hours
            now_provider=lambda: self.now,
        )

        # Setup resolver responses
        self.resolver.resolved_symbols["BTCUSDT"] = ResolvedProvider(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol="BTCUSDT",
        )
        self.resolver.resolved_symbols["SOLUSDT"] = ResolvedProvider(
            source=MarketDataSource.BYBIT_SPOT,
            provider_symbol="SOLUSDT",
        )
        self.resolver.resolved_symbols["UNKNOWN"] = None

        # By default, assume they have recent data so they don't trigger full bootstrap
        self.history.history[(MarketDataSource.BINANCE_SPOT, "BTCUSDT", "1h")] = self.now - timedelta(hours=3)
        self.history.history[(MarketDataSource.BYBIT_SPOT, "SOLUSDT", "1h")] = self.now - timedelta(hours=3)
        self._set_completed_bootstrap(MarketDataSource.BINANCE_SPOT, "BTCUSDT", "BTCUSDT")
        self._set_completed_bootstrap(MarketDataSource.BYBIT_SPOT, "SOLUSDT", "SOLUSDT")

    def _set_completed_bootstrap(self, source: MarketDataSource, canonical_symbol: str, provider_symbol: str) -> None:
        self.collection_state.states[(source, canonical_symbol, "1h")] = CandleCollectionState(
            source=source,
            canonical_symbol=canonical_symbol,
            provider_symbol=provider_symbol,
            timeframe="1h",
            bootstrap_from=self.now - timedelta(days=365 * 2),
            bootstrap_to=self.now - timedelta(days=1),
            bootstrap_next_from=self.now - timedelta(days=1),
            bootstrap_completed_at=self.now - timedelta(days=1),
            last_successful_close_time=self.now - timedelta(hours=3),
        )

    async def test_orchestrates_collection_cycle_incremental(self) -> None:
        result = await self.service.collect()

        # Check resolver calls
        self.assertEqual(self.resolver.calls, ["BTCUSDT", "SOLUSDT", "UNKNOWN"])

        # Check scheduled jobs
        self.assertEqual(result.scheduled_count, 2)
        self.assertEqual(len(self.queue.enqueued_jobs), 2)
        
        # Verify job details
        btc_job = self.queue.enqueued_jobs[0]
        self.assertEqual(btc_job.source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(btc_job.provider_symbol, "BTCUSDT")
        self.assertEqual(btc_job.timeframe, "1h")
        self.assertEqual(btc_job.job_kind, SyncJobKind.FRESH)
        self.assertEqual(btc_job.expected_close_time, datetime(2026, 7, 18, 11, tzinfo=UTC))

        sol_job = self.queue.enqueued_jobs[1]
        self.assertEqual(sol_job.source, MarketDataSource.BYBIT_SPOT)
        self.assertEqual(sol_job.provider_symbol, "SOLUSDT")
        self.assertEqual(sol_job.timeframe, "1h")
        self.assertEqual(sol_job.job_kind, SyncJobKind.FRESH)
        self.assertEqual(sol_job.expected_close_time, datetime(2026, 7, 18, 11, tzinfo=UTC))

        # Verify executing runs with create_events=True since it's FRESH job
        self.assertEqual(result.processed_count, 2)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(len(self.sync_service.calls), 2)
        self.assertTrue(self.sync_service.calls[0].create_events)
        self.assertTrue(self.sync_service.calls[1].create_events)
        self.assertEqual(self.queue.completed_keys, [btc_job.idempotency_key, sol_job.idempotency_key])

        self.assertEqual(self.publisher.calls, 0)
        self.assertEqual(result.published_count, 0)

    async def test_continues_on_sync_failure(self) -> None:
        # Make SOLUSDT sync fail
        self.sync_service.fail_for_symbol = "SOLUSDT"

        result = await self.service.collect()

        self.assertEqual(result.scheduled_count, 2)
        self.assertEqual(result.processed_count, 1)
        self.assertEqual(result.failed_count, 1)
        
        btc_job = self.queue.enqueued_jobs[0]
        sol_job = self.queue.enqueued_jobs[1]

        self.assertEqual(self.queue.completed_keys, [btc_job.idempotency_key])
        self.assertEqual(self.queue.failed_keys, [sol_job.idempotency_key])

    async def test_first_run_triggers_paginated_bootstrap(self) -> None:
        # Clean history to trigger bootstrap
        self.history.history.clear()
        self.collection_state.states.clear()

        # We set bootstrap_max_chunks_per_tick = 3
        # 2 years of 1h timeframe is ~17500 hours.
        # With chunk_limit = 100, 1 chunk is 100 hours.
        # So we should schedule exactly 3 chunks of 100 hours each.
        result = await self.service.collect()

        # 3 chunks for BTCUSDT, 3 chunks for SOLUSDT -> 6 jobs total
        self.assertEqual(result.scheduled_count, 6)
        self.assertEqual(len(self.queue.enqueued_jobs), 6)

        # Verify chunk 1 BTCUSDT
        job1 = self.queue.enqueued_jobs[0]
        self.assertEqual(job1.provider_symbol, "BTCUSDT")
        self.assertEqual(job1.job_kind, SyncJobKind.BACKFILL)
        # bootstrap start: now - 2 years (aligned to 1h)
        expected_start = self.now - timedelta(days=365 * 2)
        expected_start = expected_start.replace(minute=0, second=0, microsecond=0)
        self.assertEqual(job1.requested_from, expected_start)
        self.assertEqual(job1.requested_to, expected_start + timedelta(hours=100))

        # Verify executing runs with create_events=False since it's BACKFILL job
        self.assertEqual(result.processed_count, 6)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(len(self.sync_service.calls), 6)
        for call in self.sync_service.calls:
            self.assertFalse(call.create_events)

    async def test_resume_bootstrap_progress(self) -> None:
        # Assume BTCUSDT started bootstrap but is still 1 year behind (e.g. latest_time is now - 365 days)
        last_bootstrap_time = self.now - timedelta(days=365)
        self.collection_state.states[(MarketDataSource.BINANCE_SPOT, "BTCUSDT", "1h")] = CandleCollectionState(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTCUSDT",
            provider_symbol="BTCUSDT",
            timeframe="1h",
            bootstrap_from=self.now - timedelta(days=365 * 2),
            bootstrap_to=self.now - timedelta(hours=1),
            bootstrap_next_from=last_bootstrap_time + timedelta(hours=1),
            bootstrap_completed_at=None,
            last_successful_close_time=None,
        )
        self.history.history[(MarketDataSource.BYBIT_SPOT, "SOLUSDT", "1h")] = self.now - timedelta(hours=3) # SOL is live

        result = await self.service.collect()

        # Should schedule 3 chunks of BACKFILL for BTCUSDT, and 1 FRESH for SOLUSDT -> 4 jobs total
        self.assertEqual(result.scheduled_count, 4)
        
        # Verify BTCUSDT jobs are BACKFILL starting from last_bootstrap_time
        btc_job = self.queue.enqueued_jobs[0]
        self.assertEqual(btc_job.provider_symbol, "BTCUSDT")
        self.assertEqual(btc_job.job_kind, SyncJobKind.BACKFILL)
        self.assertEqual(btc_job.requested_from, last_bootstrap_time + timedelta(hours=1))
        self.assertEqual(btc_job.requested_to, last_bootstrap_time + timedelta(hours=101))

        # Verify SOLUSDT job is FRESH
        sol_job = self.queue.enqueued_jobs[3]
        self.assertEqual(sol_job.provider_symbol, "SOLUSDT")
        self.assertEqual(sol_job.job_kind, SyncJobKind.FRESH)

    async def test_downtime_recovery_splits_into_fresh_chunks_with_events(self) -> None:
        # Assume BTCUSDT went down for 10 days (240 hours)
        # Since we have chunk_limit = 100, 240 hours will be split into 3 chunks (max_chunks = 3):
        # Chunk 1: 100 hours (FRESH)
        # Chunk 2: 100 hours (FRESH)
        # Chunk 3: 40 hours (FRESH)
        # SOLUSDT is live (no gap)
        downtime_start = self.now - timedelta(hours=242)
        self.history.history[(MarketDataSource.BINANCE_SPOT, "BTCUSDT", "1h")] = downtime_start
        self.collection_state.states[(MarketDataSource.BINANCE_SPOT, "BTCUSDT", "1h")] = CandleCollectionState(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTCUSDT",
            provider_symbol="BTCUSDT",
            timeframe="1h",
            bootstrap_from=self.now - timedelta(days=365 * 2),
            bootstrap_to=downtime_start + timedelta(hours=1),
            bootstrap_next_from=downtime_start + timedelta(hours=1),
            bootstrap_completed_at=downtime_start,
            last_successful_close_time=downtime_start + timedelta(hours=1),
        )
        self.history.history[(MarketDataSource.BYBIT_SPOT, "SOLUSDT", "1h")] = self.now - timedelta(hours=2) # live

        result = await self.service.collect()

        # Should schedule exactly 3 FRESH jobs for BTCUSDT
        self.assertEqual(result.scheduled_count, 3)
        self.assertEqual(len(self.queue.enqueued_jobs), 3)

        # Verify job 1 details
        job1 = self.queue.enqueued_jobs[0]
        self.assertEqual(job1.provider_symbol, "BTCUSDT")
        self.assertEqual(job1.job_kind, SyncJobKind.FRESH)
        self.assertEqual(job1.requested_from, downtime_start + timedelta(hours=1))
        self.assertEqual(job1.requested_to, downtime_start + timedelta(hours=101))

        # Verify executing runs with create_events=True since they are FRESH jobs
        self.assertEqual(result.processed_count, 3)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(len(self.sync_service.calls), 3)
        for call in self.sync_service.calls:
            self.assertTrue(call.create_events)

    async def test_live_tick_scheduling_for_different_timeframes(self) -> None:
        for timeframe, duration in [("1h", timedelta(hours=1)), ("4h", timedelta(hours=4)), ("1d", timedelta(days=1))]:
            with self.subTest(timeframe=timeframe):
                queue = MockSyncJobQueue()
                history = MockCandleHistory()

                service = CandleCollectionService(
                    resolver=self.resolver,
                    single_symbol_sync=self.sync_service,
                    sync_job_queue=queue,
                    candle_history=history,
                    collection_state=MockCollectionStateRepository(),
                    provider_symbols=("BTCUSDT",),
                    timeframes=(timeframe,),
                    safety_delay_by_timeframe={timeframe: timedelta(minutes=5)},
                    jitter_seconds=0,
                    now_provider=lambda: self.now,
                )

                due_time = due_closed_candle_time(
                    now=self.now,
                    timeframe=timeframe,
                    safety_delay=timedelta(minutes=5),
                )

                latest_open_time = due_time - 2 * duration
                history.history[(MarketDataSource.BINANCE_SPOT, "BTCUSDT", timeframe)] = latest_open_time
                service.collection_state.states[(MarketDataSource.BINANCE_SPOT, "BTCUSDT", timeframe)] = CandleCollectionState(
                    source=MarketDataSource.BINANCE_SPOT,
                    canonical_symbol="BTCUSDT",
                    provider_symbol="BTCUSDT",
                    timeframe=timeframe,
                    bootstrap_from=self.now - timedelta(days=365 * 2),
                    bootstrap_to=due_time - 10 * duration,
                    bootstrap_next_from=due_time - 10 * duration,
                    bootstrap_completed_at=self.now - timedelta(days=1),
                    last_successful_close_time=latest_open_time + duration,
                )

                result = await service.collect()

                self.assertEqual(result.scheduled_count, 1)
                self.assertEqual(len(queue.enqueued_jobs), 1)

                job = queue.enqueued_jobs[0]
                self.assertEqual(job.provider_symbol, "BTCUSDT")
                self.assertEqual(job.timeframe, timeframe)
                self.assertEqual(job.job_kind, SyncJobKind.FRESH)
                self.assertEqual(job.requested_from, due_time - duration)
                self.assertEqual(job.requested_to, due_time)

    async def test_concurrency_limiter_is_acquired_during_execution(self) -> None:
        limiter = MockConcurrencyLimiter()

        service = CandleCollectionService(
            resolver=self.resolver,
            single_symbol_sync=self.sync_service,
            sync_job_queue=self.queue,
            candle_history=self.history,
            collection_state=self.collection_state,
            provider_symbols=("BTCUSDT", "SOLUSDT"),
            timeframes=("1h",),
            safety_delay_by_timeframe={"1h": timedelta(minutes=5)},
            jitter_seconds=0,
            concurrency_limiter=limiter,
            now_provider=lambda: self.now,
        )

        result = await service.collect()

        self.assertEqual(result.processed_count, 2)
        self.assertEqual(limiter.entered, 2)
        self.assertEqual(limiter.exited, 2)


class MockConcurrencyLimiter:
    def __init__(self) -> None:
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> "MockConcurrencyLimiter":
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.exited += 1
        return False
