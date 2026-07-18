from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import CandleRangeStatus, MarketDataSource
from market_data_service.domain.events.candle_batch_ready import CandleBatchReady
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.domain.snapshot_models import MarketSnapshot
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository


class FakeExecuteResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class FakeConnection:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeExecuteResult(self.rowcount)


class OutboxRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_pending_candle_batch_ready_reports_inserted_event(self) -> None:
        connection = FakeConnection(rowcount=1)
        repository = OutboxRepository(connection)

        created = await repository.create_pending_candle_batch_ready(CandleBatchReady.from_snapshot(_snapshot()))

        self.assertTrue(created)
        self.assertEqual(len(connection.statements), 1)

    async def test_create_pending_candle_batch_ready_reports_duplicate_event(self) -> None:
        connection = FakeConnection(rowcount=0)
        repository = OutboxRepository(connection)

        created = await repository.create_pending_candle_batch_ready(CandleBatchReady.from_snapshot(_snapshot()))

        self.assertFalse(created)
        self.assertEqual(len(connection.statements), 1)

    async def test_create_pending_market_data_backfill_requested_reports_inserted_event(self) -> None:
        connection = FakeConnection(rowcount=1)
        repository = OutboxRepository(connection)

        created = await repository.create_pending_market_data_backfill_requested(_backfill_event())

        self.assertTrue(created)
        self.assertEqual(len(connection.statements), 1)

    async def test_delete_old_published_events_runs_in_batches_until_empty(self) -> None:
        connection = FakeConnectionForCleanup(select_results=[[("id-1",), ("id-2",)]])
        repository = OutboxRepository(connection)

        cutoff = datetime(2026, 7, 11, 12, 0, 0, tzinfo=UTC)
        deleted = await repository.delete_old_published_events(cutoff=cutoff, batch_size=2)

        self.assertEqual(deleted, 2)
        self.assertEqual(len(connection.statements), 3)


class FakeCleanupExecuteResult:
    def __init__(self, rows: list[tuple[str]]) -> None:
        self.rows = rows

    def all(self) -> list[tuple[str]]:
        return self.rows


class FakeConnectionForCleanup:
    def __init__(self, select_results: list[list[tuple[str]]]) -> None:
        self.select_results = select_results
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        stmt_str = str(statement).lower()
        if "select" in stmt_str:
            if self.select_results:
                return FakeCleanupExecuteResult(self.select_results.pop(0))
            return FakeCleanupExecuteResult([])
        return FakeExecuteResult(1)


def _snapshot() -> MarketSnapshot:
    return MarketSnapshot(
        id="snapshot-1",
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


def _backfill_event() -> MarketDataBackfillRequested:
    return MarketDataBackfillRequested.from_gap(
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol="ETH/USDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        requested_from=datetime(2026, 7, 14, 9, tzinfo=UTC),
        requested_to=datetime(2026, 7, 14, 10, tzinfo=UTC),
        parent_batch_id="batch-1",
        occurred_at=datetime(2026, 7, 14, 10, 1, tzinfo=UTC),
    )
