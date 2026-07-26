from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from market_data_service.application.services.candle_collection_service import CandleCollectionService
from market_data_service.application.services.outbox_publisher_service import OutboxPublishBatchResult
from market_data_service.application.sync_models import SyncClosedCandlesCommand, SyncClosedCandlesResult
from market_data_service.domain.collection_models import CandleCollectionState
from market_data_service.domain.enums import CandleRangeStatus, MarketDataBatchStatus, MarketDataSource, SyncJobKind
from market_data_service.domain.scheduler_models import MarketDataSyncJob
from market_data_service.domain.timeframe_rules import get_timeframe_duration
from market_data_service.workers.market_data_scheduler_worker import MarketDataSchedulerWorker


class E2EMultiProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_e2e_multi_provider_candle_collection_pipeline(self) -> None:
        base_time = datetime(2026, 7, 18, 19, 0, 0, tzinfo=UTC)
        queue = FakeSyncJobQueue()
        sync_service = FakeSingleSymbolSyncService()
        state = FakeCollectionStateRepository()
        service = CandleCollectionService(
            resolver=FakeResolver(),
            single_symbol_sync=sync_service,
            sync_job_queue=queue,
            candle_history=FakeCandleHistory(),
            collection_state=state,
            provider_symbols=("BTC/USDT",),
            timeframes=("1h",),
            safety_delay_by_timeframe={"1h": timedelta(0)},
            jitter_seconds=0,
            bootstrap_lookback_years=2,
            bootstrap_max_chunks_per_tick=3,
            chunk_limit=5,
            now_provider=lambda: base_time,
        )
        worker = MarketDataSchedulerWorker(service, poll_interval_seconds=0.01)

        bootstrap_result = await worker.run_once()

        self.assertEqual(bootstrap_result.scheduled_count, 3)
        self.assertEqual(bootstrap_result.processed_count, 3)
        self.assertEqual(bootstrap_result.failed_count, 0)
        self.assertEqual({job.job_kind for job in queue.enqueued_jobs}, {SyncJobKind.BACKFILL})
        self.assertTrue(sync_service.calls)
        self.assertTrue(all(not call.create_events for call in sync_service.calls))

        bootstrap_to = base_time.replace(minute=0, second=0, microsecond=0)
        state.states[(MarketDataSource.BINANCE_SPOT, "BTC/USDT", "1h")] = CandleCollectionState(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTC/USDT",
            provider_symbol="BTCUSDT",
            timeframe="1h",
            bootstrap_from=base_time - timedelta(days=365 * 2),
            bootstrap_to=bootstrap_to,
            bootstrap_next_from=bootstrap_to,
            bootstrap_completed_at=base_time,
            last_successful_close_time=bootstrap_to,
        )
        queue.clear()
        sync_service.calls.clear()
        service.now_provider = lambda: base_time + timedelta(hours=2, minutes=10)

        incremental_result = await worker.run_once()

        self.assertEqual(incremental_result.scheduled_count, 1)
        self.assertEqual(incremental_result.processed_count, 1)
        self.assertEqual(queue.enqueued_jobs[0].job_kind, SyncJobKind.FRESH)
        self.assertEqual(len(sync_service.calls), 1)
        self.assertTrue(sync_service.calls[0].create_events)

        outbox = FakeOutboxPublisher()
        publish_result = await outbox.run_once()

        self.assertEqual(publish_result.fetched_count, 2)
        self.assertEqual(publish_result.published_count, 2)


class FakeResolver:
    async def resolve(self, symbol: str):
        return SimpleNamespace(source=MarketDataSource.BINANCE_SPOT, provider_symbol=symbol.replace("/", ""))


class FakeCandleHistory:
    async def get_latest_closed_candle_time(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ):
        return None


class FakeCollectionStateRepository:
    def __init__(self) -> None:
        self.states: dict[tuple[MarketDataSource, str, str], CandleCollectionState] = {}

    async def get(self, *, source: MarketDataSource, canonical_symbol: str, timeframe: str):
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
            last_successful_close_time=last_successful_close_time,
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


class FakeSyncJobQueue:
    def __init__(self) -> None:
        self.enqueued_jobs: list[MarketDataSyncJob] = []
        self.pending_jobs: list[MarketDataSyncJob] = []
        self.completed_keys: list[str] = []

    async def enqueue_sync_job(self, job: MarketDataSyncJob) -> bool:
        self.enqueued_jobs.append(job)
        self.pending_jobs.append(job)
        return True

    async def claim_next_pending_job(self, *, now: datetime) -> MarketDataSyncJob | None:
        if not self.pending_jobs:
            return None
        job = self.pending_jobs.pop(0)
        return MarketDataSyncJob(
            source=job.source,
            provider_symbol=job.provider_symbol,
            timeframe=job.timeframe,
            expected_close_time=job.expected_close_time,
            scheduled_for=job.scheduled_for,
            idempotency_key=job.idempotency_key,
            priority=job.priority,
            job_kind=job.job_kind,
            requested_from=job.requested_from,
            requested_to=job.requested_to,
        )

    async def mark_completed(self, *, idempotency_key: str, completed_at: datetime) -> None:
        self.completed_keys.append(idempotency_key)

    async def mark_failed(self, *, idempotency_key: str, completed_at: datetime) -> None:
        raise AssertionError("Smoke pipeline should not fail jobs")

    def clear(self) -> None:
        self.enqueued_jobs.clear()
        self.pending_jobs.clear()
        self.completed_keys.clear()


class FakeSingleSymbolSyncService:
    def __init__(self) -> None:
        self.calls: list[SyncClosedCandlesCommand] = []

    async def sync_closed_candles(self, command: SyncClosedCandlesCommand) -> SyncClosedCandlesResult:
        self.calls.append(command)
        duration = get_timeframe_duration(command.timeframe)
        fetched_count = max(1, int((command.to_time - command.from_time) / duration))
        return SyncClosedCandlesResult(
            batch_id=f"batch-{len(self.calls)}",
            source=command.source,
            canonical_symbol=command.canonical_symbol or command.provider_symbol,
            provider_symbol=command.provider_symbol,
            timeframe=command.timeframe,
            fetched_count=fetched_count,
            inserted_count=fetched_count,
            skipped_duplicate_count=0,
            range_status=CandleRangeStatus.COMPLETE,
            batch_status=MarketDataBatchStatus.COMPLETE,
            gap_count=0,
            dry_run=False,
            snapshot_id=f"snapshot-{len(self.calls)}",
            snapshot_created=True,
        )


class FakeOutboxPublisher:
    async def run_once(self) -> OutboxPublishBatchResult:
        return OutboxPublishBatchResult(
            fetched_count=2,
            published_count=2,
            retry_count=0,
            failed_count=0,
            publisher_lag_seconds=0.0,
        )
