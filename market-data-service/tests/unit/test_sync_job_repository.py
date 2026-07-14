from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.domain.scheduler_models import MarketDataSyncJob
from market_data_service.persistence.repositories.sync_job_repository import SyncJobRepository


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


class SyncJobRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_sync_job_reports_created_job(self) -> None:
        connection = FakeConnection(rowcount=1)
        repository = SyncJobRepository(connection)

        created = await repository.enqueue_sync_job(_job())

        self.assertTrue(created)
        self.assertEqual(len(connection.statements), 1)

    async def test_enqueue_sync_job_reports_duplicate_job(self) -> None:
        connection = FakeConnection(rowcount=0)
        repository = SyncJobRepository(connection)

        created = await repository.enqueue_sync_job(_job())

        self.assertFalse(created)
        self.assertEqual(len(connection.statements), 1)

    async def test_enqueue_backfill_job_uses_backfill_event(self) -> None:
        connection = FakeConnection(rowcount=1)
        repository = SyncJobRepository(connection)

        created = await repository.enqueue_backfill_job(_backfill_event())

        self.assertTrue(created)
        self.assertEqual(len(connection.statements), 1)


def _job() -> MarketDataSyncJob:
    return MarketDataSyncJob(
        source=MarketDataSource.BINANCE_SPOT,
        provider_symbol="ETHUSDT",
        timeframe="1h",
        expected_close_time=datetime(2026, 7, 14, 10, tzinfo=UTC),
        scheduled_for=datetime(2026, 7, 14, 10, 0, 30, tzinfo=UTC),
        idempotency_key="BINANCE_SPOT|ETHUSDT|1h|2026-07-14T10:00:00+00:00",
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
