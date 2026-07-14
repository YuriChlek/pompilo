from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import unittest

from market_data_service.application.services.single_symbol_sync_service import (
    SingleSymbolSyncService,
    SyncAlreadyRunningError,
)
from market_data_service.application.services.backfill_planning_service import BackfillPlanningResult
from market_data_service.application.services.symbol_registry_service import ProviderSymbolNotActiveError
from market_data_service.application.sync_models import SyncClosedCandlesCommand
from market_data_service.domain.batch_models import MarketDataBatch
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import (
    CandleRangeStatus,
    MarketDataBatchStatus,
    MarketDataSource,
    OutboxStatus,
    ProviderSymbolStatus,
)
from market_data_service.domain.snapshot_models import MarketSnapshot, SnapshotCreationResult
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.observability.metrics import MARKET_DATA_GAP_COUNT, InMemoryMetricsRecorder
from market_data_service.observability.structured_logging import InMemoryStructuredLogger


def _provider_symbol() -> ProviderSymbol:
    return ProviderSymbol(
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        status=ProviderSymbolStatus.TRADING,
        supported_timeframes=("1h", "4h", "1d"),
    )


def _candle(open_time: datetime) -> CanonicalCandle:
    return CanonicalCandle(
        candle_id="candle-1",
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        open_time=open_time,
        close_time=open_time + timedelta(hours=1),
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("90"),
        close=Decimal("105"),
        volume=Decimal("10"),
        quote_volume=Decimal("1050"),
        taker_buy_base_volume=Decimal("6"),
        taker_buy_quote_volume=Decimal("630"),
        taker_sell_base_volume=Decimal("4"),
        taker_sell_quote_volume=Decimal("420"),
        trades_count=10,
        is_closed=True,
        provider_payload_hash="hash",
    )


class FakeRegistry:
    def __init__(self, mapping: ProviderSymbol | None) -> None:
        self.mapping = mapping

    async def get_provider_symbol(self, source: MarketDataSource, provider_symbol: str) -> ProviderSymbol | None:
        return self.mapping


class FakeProvider:
    def __init__(self, candles: list[CanonicalCandle]) -> None:
        self.candles = candles
        self.calls = 0
        self.error: Exception | None = None

    async def fetch_closed_candles(self, provider_symbol, *, timeframe: str, from_time: datetime, to_time: datetime):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.candles


class FakeWriter:
    def __init__(self, inserted_count: int) -> None:
        self.inserted_count = inserted_count
        self.received: list[CanonicalCandle] | None = None
        self.error: Exception | None = None

    async def insert_closed_candles(self, candles: list[CanonicalCandle]) -> int:
        if self.error is not None:
            raise self.error
        self.received = candles
        return self.inserted_count


class FakeLock:
    def __init__(self, acquired: bool = True) -> None:
        self.acquired = acquired
        self.keys: list[tuple[str, str, str]] = []

    async def acquire_sync_lock(self, *, source: str, provider_symbol: str, timeframe: str) -> bool:
        self.keys.append((source, provider_symbol, timeframe))
        return self.acquired


class FakeBatchTracker:
    def __init__(self) -> None:
        self.created: list[MarketDataBatch] = []
        self.completed: list[dict[str, object]] = []
        self.failed: list[dict[str, str]] = []

    async def create_running_batch(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        requested_from: datetime,
        requested_to: datetime,
        expected_close_time: datetime,
    ) -> MarketDataBatch:
        batch = MarketDataBatch(
            batch_id=f"batch-{len(self.created) + 1}",
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            requested_from=requested_from,
            requested_to=requested_to,
            expected_close_time=expected_close_time,
            status=MarketDataBatchStatus.RUNNING,
            outbox_status=OutboxStatus.NOT_CREATED,
            rows_fetched=0,
            rows_inserted=0,
            rows_skipped_duplicate=0,
            rows_hash_mismatch=0,
            gap_count=0,
            first_open_time=None,
            last_close_time=None,
            error_code=None,
            error_message_redacted=None,
            started_at=datetime(2026, 7, 14, 8, tzinfo=UTC),
            completed_at=None,
        )
        self.created.append(batch)
        return batch

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
    ) -> None:
        self.completed.append(
            {
                "batch_id": batch_id,
                "status": status,
                "rows_fetched": rows_fetched,
                "rows_inserted": rows_inserted,
                "rows_skipped_duplicate": rows_skipped_duplicate,
                "rows_hash_mismatch": rows_hash_mismatch,
                "gap_count": gap_count,
                "first_open_time": first_open_time,
                "last_close_time": last_close_time,
            }
        )

    async def fail_batch(self, *, batch_id: str, error_code: str, error_message_redacted: str) -> None:
        self.failed.append(
            {
                "batch_id": batch_id,
                "error_code": error_code,
                "error_message_redacted": error_message_redacted,
            }
        )


class FakeSyncCompletion:
    def __init__(
        self,
        *,
        inserted_count: int,
        skipped_duplicate_count: int = 0,
        snapshot_result: SnapshotCreationResult | None = None,
    ) -> None:
        self.inserted_count = inserted_count
        self.skipped_duplicate_count = skipped_duplicate_count
        self.snapshot_result = snapshot_result
        self.calls: list[dict[str, object]] = []

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
    ) -> tuple[int, int, SnapshotCreationResult | None]:
        self.calls.append(
            {
                "batch_id": batch_id,
                "source": source,
                "canonical_symbol": canonical_symbol,
                "timeframe": timeframe,
                "candles": candles,
                "batch_status": batch_status,
                "rows_fetched": rows_fetched,
                "dry_run": dry_run,
                "gap_count": gap_count,
                "first_open_time": first_open_time,
                "last_close_time": last_close_time,
            }
        )
        return self.inserted_count, self.skipped_duplicate_count, self.snapshot_result


class FakeBackfillPlanning:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def request_backfill_for_gaps(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        parent_batch_id: str,
        missing_intervals,
    ) -> BackfillPlanningResult:
        self.calls.append(
            {
                "source": source,
                "canonical_symbol": canonical_symbol,
                "provider_symbol": provider_symbol,
                "timeframe": timeframe,
                "parent_batch_id": parent_batch_id,
                "missing_intervals": missing_intervals,
            }
        )
        return BackfillPlanningResult(requested_count=len(missing_intervals), skipped_duplicate_count=0)


class SingleSymbolSyncServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_fetches_and_inserts_closed_candles(self) -> None:
        candles = [_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))]
        writer = FakeWriter(inserted_count=1)
        lock = FakeLock()
        batch_tracker = FakeBatchTracker()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider(candles),
            candle_writer=writer,
            advisory_lock=lock,
            batch_tracker=batch_tracker,
        )

        result = await service.sync_closed_candles(_command())

        self.assertEqual(result.fetched_count, 1)
        self.assertEqual(result.inserted_count, 1)
        self.assertEqual(result.skipped_duplicate_count, 0)
        self.assertEqual(result.range_status, CandleRangeStatus.COMPLETE)
        self.assertEqual(result.batch_id, "batch-1")
        self.assertEqual(result.batch_status, MarketDataBatchStatus.COMPLETE)
        self.assertEqual(result.gap_count, 0)
        self.assertEqual(writer.received, candles)
        self.assertEqual(lock.keys, [("BINANCE_SPOT", "ETHUSDT", "1h")])
        self.assertEqual(len(batch_tracker.created), 1)
        self.assertEqual(batch_tracker.completed[0]["status"], MarketDataBatchStatus.COMPLETE)
        self.assertEqual(batch_tracker.completed[0]["first_open_time"], datetime(2026, 7, 14, 8, tzinfo=UTC))
        self.assertEqual(batch_tracker.completed[0]["last_close_time"], datetime(2026, 7, 14, 9, tzinfo=UTC))

    async def test_sync_reports_skipped_duplicates_from_insert_count(self) -> None:
        candles = [
            _candle(datetime(2026, 7, 14, 8, tzinfo=UTC)),
            _candle(datetime(2026, 7, 14, 9, tzinfo=UTC)),
        ]
        batch_tracker = FakeBatchTracker()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider(candles),
            candle_writer=FakeWriter(inserted_count=1),
            advisory_lock=FakeLock(),
            batch_tracker=batch_tracker,
        )

        result = await service.sync_closed_candles(_command())

        self.assertEqual(result.fetched_count, 2)
        self.assertEqual(result.inserted_count, 1)
        self.assertEqual(result.skipped_duplicate_count, 1)
        self.assertEqual(result.range_status, CandleRangeStatus.COMPLETE)
        self.assertEqual(batch_tracker.completed[0]["rows_skipped_duplicate"], 1)

    async def test_sync_returns_gap_status_for_incomplete_range(self) -> None:
        batch_tracker = FakeBatchTracker()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider([_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))]),
            candle_writer=FakeWriter(inserted_count=1),
            advisory_lock=FakeLock(),
            batch_tracker=batch_tracker,
        )

        result = await service.sync_closed_candles(
            SyncClosedCandlesCommand(
                source=MarketDataSource.BINANCE_SPOT,
                provider_symbol="ETHUSDT",
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
            )
        )

        self.assertEqual(result.range_status, CandleRangeStatus.INCOMPLETE)
        self.assertEqual(result.batch_status, MarketDataBatchStatus.INCOMPLETE)
        self.assertEqual(result.gap_count, 1)
        self.assertEqual(batch_tracker.completed[0]["status"], MarketDataBatchStatus.INCOMPLETE)

    async def test_sync_requests_backfill_for_internal_gap(self) -> None:
        backfill_planning = FakeBackfillPlanning()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider(
                [
                    _candle(datetime(2026, 7, 14, 8, tzinfo=UTC)),
                    _candle(datetime(2026, 7, 14, 10, tzinfo=UTC)),
                ]
            ),
            candle_writer=FakeWriter(inserted_count=2),
            advisory_lock=FakeLock(),
            batch_tracker=FakeBatchTracker(),
            backfill_planning=backfill_planning,
        )

        result = await service.sync_closed_candles(
            SyncClosedCandlesCommand(
                source=MarketDataSource.BINANCE_SPOT,
                provider_symbol="ETHUSDT",
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 11, tzinfo=UTC),
            )
        )

        self.assertEqual(result.range_status, CandleRangeStatus.GAP_DETECTED)
        self.assertEqual(len(backfill_planning.calls), 1)
        self.assertEqual(backfill_planning.calls[0]["provider_symbol"], "ETHUSDT")
        self.assertEqual(backfill_planning.calls[0]["parent_batch_id"], "batch-1")
        self.assertEqual(backfill_planning.calls[0]["missing_intervals"][0].open_time, datetime(2026, 7, 14, 9, tzinfo=UTC))

    async def test_sync_does_not_request_backfill_for_latest_incomplete_range(self) -> None:
        backfill_planning = FakeBackfillPlanning()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider([_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))]),
            candle_writer=FakeWriter(inserted_count=1),
            advisory_lock=FakeLock(),
            batch_tracker=FakeBatchTracker(),
            backfill_planning=backfill_planning,
        )

        result = await service.sync_closed_candles(
            SyncClosedCandlesCommand(
                source=MarketDataSource.BINANCE_SPOT,
                provider_symbol="ETHUSDT",
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
            )
        )

        self.assertEqual(result.range_status, CandleRangeStatus.INCOMPLETE)
        self.assertEqual(backfill_planning.calls, [])

    async def test_dry_run_fetches_but_does_not_insert(self) -> None:
        writer = FakeWriter(inserted_count=1)
        batch_tracker = FakeBatchTracker()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider([_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))]),
            candle_writer=writer,
            advisory_lock=FakeLock(),
            batch_tracker=batch_tracker,
        )

        result = await service.sync_closed_candles(_command(dry_run=True))

        self.assertTrue(result.dry_run)
        self.assertEqual(result.inserted_count, 0)
        self.assertEqual(result.skipped_duplicate_count, 0)
        self.assertIsNone(writer.received)
        self.assertEqual(batch_tracker.completed[0]["rows_inserted"], 0)

    async def test_sync_blocks_missing_mapping(self) -> None:
        batch_tracker = FakeBatchTracker()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(None),
            candle_provider=FakeProvider([]),
            candle_writer=FakeWriter(inserted_count=0),
            advisory_lock=FakeLock(),
            batch_tracker=batch_tracker,
        )

        with self.assertRaises(ProviderSymbolNotActiveError):
            await service.sync_closed_candles(_command())
        self.assertEqual(batch_tracker.created, [])

    async def test_sync_blocks_parallel_duplicate_lock(self) -> None:
        batch_tracker = FakeBatchTracker()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider([]),
            candle_writer=FakeWriter(inserted_count=0),
            advisory_lock=FakeLock(acquired=False),
            batch_tracker=batch_tracker,
        )

        with self.assertRaises(SyncAlreadyRunningError):
            await service.sync_closed_candles(_command())
        self.assertEqual(batch_tracker.created, [])

    async def test_sync_marks_batch_failed_when_provider_raises(self) -> None:
        provider = FakeProvider([])
        provider.error = RuntimeError("provider unavailable")
        batch_tracker = FakeBatchTracker()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=provider,
            candle_writer=FakeWriter(inserted_count=0),
            advisory_lock=FakeLock(),
            batch_tracker=batch_tracker,
        )

        with self.assertRaises(RuntimeError):
            await service.sync_closed_candles(_command())

        self.assertEqual(batch_tracker.completed, [])
        self.assertEqual(batch_tracker.failed[0]["batch_id"], "batch-1")
        self.assertEqual(batch_tracker.failed[0]["error_code"], "RuntimeError")

    async def test_sync_completion_creates_snapshot_for_complete_non_dry_run_sync(self) -> None:
        candles = [_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))]
        snapshot = _snapshot(snapshot_id="snapshot-1")
        sync_completion = FakeSyncCompletion(
            inserted_count=1,
            snapshot_result=SnapshotCreationResult(snapshot=snapshot, created=True),
        )
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider(candles),
            candle_writer=FakeWriter(inserted_count=0),
            advisory_lock=FakeLock(),
            batch_tracker=FakeBatchTracker(),
            sync_completion=sync_completion,
        )

        result = await service.sync_closed_candles(_command())

        self.assertEqual(result.snapshot_id, "snapshot-1")
        self.assertTrue(result.snapshot_created)
        self.assertEqual(sync_completion.calls[0]["batch_status"], MarketDataBatchStatus.COMPLETE)
        self.assertEqual(sync_completion.calls[0]["canonical_symbol"], "ETH/USDT")

    async def test_sync_completion_does_not_return_snapshot_for_incomplete_range(self) -> None:
        sync_completion = FakeSyncCompletion(inserted_count=1)
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider([_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))]),
            candle_writer=FakeWriter(inserted_count=0),
            advisory_lock=FakeLock(),
            batch_tracker=FakeBatchTracker(),
            sync_completion=sync_completion,
        )

        result = await service.sync_closed_candles(
            SyncClosedCandlesCommand(
                source=MarketDataSource.BINANCE_SPOT,
                provider_symbol="ETHUSDT",
                timeframe="1h",
                from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
                to_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
            )
        )

        self.assertIsNone(result.snapshot_id)
        self.assertFalse(result.snapshot_created)
        self.assertEqual(sync_completion.calls[0]["batch_status"], MarketDataBatchStatus.INCOMPLETE)

    async def test_sync_emits_metrics_and_structured_log_with_correlation_id(self) -> None:
        metrics = InMemoryMetricsRecorder()
        logs = InMemoryStructuredLogger()
        service = SingleSymbolSyncService(
            symbol_registry=FakeRegistry(_provider_symbol()),
            candle_provider=FakeProvider([_candle(datetime(2026, 7, 14, 8, tzinfo=UTC))]),
            candle_writer=FakeWriter(inserted_count=1),
            advisory_lock=FakeLock(),
            batch_tracker=FakeBatchTracker(),
            metrics_recorder=metrics,
            structured_logger=logs,
            now_provider=lambda: datetime(2026, 7, 14, 10, tzinfo=UTC),
        )

        result = await service.sync_closed_candles(_command(correlation_id="corr-sync"))

        self.assertEqual(result.batch_status, MarketDataBatchStatus.COMPLETE)
        self.assertIn(MARKET_DATA_GAP_COUNT, {sample.name for sample in metrics.samples})
        self.assertEqual(logs.events[0].fields["batch_id"], "batch-1")
        self.assertEqual(logs.events[0].fields["correlation_id"], "corr-sync")


def _command(*, dry_run: bool = False, correlation_id: str | None = None) -> SyncClosedCandlesCommand:
    return SyncClosedCandlesCommand(
        source=MarketDataSource.BINANCE_SPOT,
        provider_symbol="ethusdt",
        timeframe="1H",
        from_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
        to_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        dry_run=dry_run,
        correlation_id=correlation_id,
    )


def _snapshot(*, snapshot_id: str) -> MarketSnapshot:
    return MarketSnapshot(
        id=snapshot_id,
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        timeframe="1h",
        last_closed_candle_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        lookback_start_time=datetime(2026, 7, 14, 8, tzinfo=UTC),
        lookback_end_time=datetime(2026, 7, 14, 9, tzinfo=UTC),
        candle_count=1,
        data_hash="data-hash",
        batch_id="batch-1",
        completeness_status=CandleRangeStatus.COMPLETE,
        snapshot_version=1,
        created_at=datetime(2026, 7, 14, 9, tzinfo=UTC),
    )
