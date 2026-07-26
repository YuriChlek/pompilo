from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import unittest
from unittest.mock import AsyncMock, MagicMock

from market_data_service.domain.enums import CandleRangeStatus, MarketDataBatchStatus, MarketDataSource
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.snapshot_models import SnapshotCreationResult, MarketSnapshot
from market_data_service.persistence.repositories.sync_completion_repository import SyncCompletionRepository


class FakeTransaction:
    def __init__(self) -> None:
        self.committed = False
        self.rolled_back = False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


class FakeConnectionForTransaction:
    def __init__(self) -> None:
        self._in_transaction = False
        self.transaction = FakeTransaction()

    def in_transaction(self) -> bool:
        return self._in_transaction

    @asynccontextmanager
    async def begin(self):
        self._in_transaction = True
        try:
            yield self.transaction
            if not self.transaction.rolled_back:
                await self.transaction.commit()
        except Exception:
            await self.transaction.rollback()
            raise
        finally:
            self._in_transaction = False


class TransactionalOutboxTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.connection = FakeConnectionForTransaction()
        self.repository = SyncCompletionRepository(self.connection)

        # Mock the underlying sub-repositories
        self.repository.candle_repository = MagicMock()
        self.repository.candle_repository.insert_closed_candles = AsyncMock(return_value=1)

        self.repository.batch_repository = MagicMock()
        self.repository.batch_repository.complete_batch = AsyncMock()

        self.snapshot = MarketSnapshot(
            id="snapshot-123",
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTC/USDT",
            timeframe="1h",
            last_closed_candle_time=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
            lookback_start_time=datetime(2026, 7, 16, 0, 0, 0, tzinfo=UTC),
            lookback_end_time=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
            candle_count=1,
            data_hash="abc",
            batch_id="batch-456",
            completeness_status=CandleRangeStatus.COMPLETE,
            snapshot_version=1,
            created_at=datetime(2026, 7, 16, 1, 5, 0, tzinfo=UTC),
        )
        self.repository.snapshot_repository = MagicMock()
        self.repository.snapshot_repository.create_snapshot_if_changed = AsyncMock(
            return_value=SnapshotCreationResult(snapshot=self.snapshot, created=True)
        )

        self.repository.outbox_repository = MagicMock()
        self.repository.outbox_repository.create_pending_candle_batch_ready = AsyncMock(return_value=True)
        self.repository.outbox_repository.create_pending_market_data_candles_collected = AsyncMock(return_value=True)

    async def test_successful_sync_commits_transaction_and_writes_outbox(self) -> None:
        inserted_count, skipped_count, snapshot_result = await self.repository.complete_sync(
            batch_id="batch-456",
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTC/USDT",
            timeframe="1h",
            candles=[],
            batch_status=MarketDataBatchStatus.COMPLETE,
            rows_fetched=1,
            dry_run=False,
            create_events=True,
            gap_count=0,
            first_open_time=datetime(2026, 7, 16, 0, 0, 0, tzinfo=UTC),
            last_close_time=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
        )

        self.assertEqual(inserted_count, 1)
        self.assertTrue(self.connection.transaction.committed)
        self.assertFalse(self.connection.transaction.rolled_back)
        self.repository.outbox_repository.create_pending_market_data_candles_collected.assert_called_once()

    async def test_successful_sync_without_create_events_does_not_write_outbox(self) -> None:
        inserted_count, skipped_count, snapshot_result = await self.repository.complete_sync(
            batch_id="batch-456",
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTC/USDT",
            timeframe="1h",
            candles=[],
            batch_status=MarketDataBatchStatus.COMPLETE,
            rows_fetched=1,
            dry_run=False,
            create_events=False,
            gap_count=0,
            first_open_time=datetime(2026, 7, 16, 0, 0, 0, tzinfo=UTC),
            last_close_time=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
        )

        self.assertEqual(inserted_count, 1)
        self.assertEqual(skipped_count, -1)
        self.assertIs(snapshot_result.snapshot, self.snapshot)
        self.assertTrue(self.connection.transaction.committed)
        self.repository.outbox_repository.create_pending_candle_batch_ready.assert_not_called()
        self.repository.outbox_repository.create_pending_market_data_candles_collected.assert_not_called()

    async def test_failed_db_write_rolls_back_transaction_and_does_not_commit_outbox(self) -> None:
        # Simulate database failure during snapshot creation
        self.repository.snapshot_repository.create_snapshot_if_changed.side_effect = RuntimeError("DB write failed")

        with self.assertRaises(RuntimeError):
            await self.repository.complete_sync(
                batch_id="batch-456",
                source=MarketDataSource.BINANCE_SPOT,
                canonical_symbol="BTC/USDT",
                timeframe="1h",
                candles=[],
                batch_status=MarketDataBatchStatus.COMPLETE,
                rows_fetched=1,
                dry_run=False,
                create_events=True,
                gap_count=0,
                first_open_time=datetime(2026, 7, 16, 0, 0, 0, tzinfo=UTC),
                last_close_time=datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC),
            )

        self.assertFalse(self.connection.transaction.committed)
        self.assertTrue(self.connection.transaction.rolled_back)
        self.repository.outbox_repository.create_pending_market_data_candles_collected.assert_not_called()
