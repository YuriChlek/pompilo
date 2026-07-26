from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested
from market_data_service.persistence.repositories.backfill_request_repository import BackfillRequestRepository


class FakeExecuteResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class FakeConnection:
    def __init__(self, rowcounts: tuple[int, ...]) -> None:
        self.rowcounts = list(rowcounts)
        self.statements: list[object] = []

    def in_transaction(self) -> bool:
        return True

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeExecuteResult(self.rowcounts.pop(0))


class BackfillRequestRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_request_backfill_creates_sync_job_and_outbox_event(self) -> None:
        connection = FakeConnection(rowcounts=(1, 1))
        repository = BackfillRequestRepository(connection)

        created = await repository.request_backfill(_event())

        self.assertTrue(created)
        self.assertEqual(len(connection.statements), 2)

    async def test_request_backfill_skips_outbox_event_for_duplicate_sync_job(self) -> None:
        connection = FakeConnection(rowcounts=(0,))
        repository = BackfillRequestRepository(connection)

        created = await repository.request_backfill(_event())

        self.assertFalse(created)
        self.assertEqual(len(connection.statements), 1)


def _event() -> MarketDataBackfillRequested:
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
