from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from market_data_service.domain.collection_models import CandleCollectionState
from market_data_service.domain.enums import MarketDataSource
from market_data_service.persistence.repositories.collection_state_repository import CollectionStateRepository


class FakeScalarResult:
    def __init__(self, row) -> None:
        self.row = row

    def one_or_none(self):
        return self.row


class FakeExecuteResult:
    def __init__(self, row=None) -> None:
        self.row = row

    def mappings(self) -> FakeScalarResult:
        return FakeScalarResult(self.row)


class FakeConnection:
    def __init__(self, row=None) -> None:
        self.row = row
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeExecuteResult(self.row)


class CollectionStateRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_returns_collection_state(self) -> None:
        row = {
            "source": "BINANCE_SPOT",
            "canonical_symbol": "BTCUSDT",
            "provider_symbol": "BTCUSDT",
            "timeframe": "1h",
            "bootstrap_from": datetime(2024, 7, 18, tzinfo=UTC),
            "bootstrap_to": datetime(2026, 7, 18, 11, tzinfo=UTC),
            "bootstrap_next_from": datetime(2024, 7, 22, tzinfo=UTC),
            "bootstrap_completed_at": None,
            "last_successful_close_time": None,
        }
        repository = CollectionStateRepository(FakeConnection(row=row))

        state = await repository.get(source=MarketDataSource.BINANCE_SPOT, canonical_symbol="BTCUSDT", timeframe="1h")

        self.assertEqual(
            state,
            CandleCollectionState(
                source=MarketDataSource.BINANCE_SPOT,
                canonical_symbol="BTCUSDT",
                provider_symbol="BTCUSDT",
                timeframe="1h",
                bootstrap_from=row["bootstrap_from"],
                bootstrap_to=row["bootstrap_to"],
                bootstrap_next_from=row["bootstrap_next_from"],
                bootstrap_completed_at=None,
                last_successful_close_time=None,
            ),
        )

    async def test_mark_progress_uses_update_statement(self) -> None:
        connection = FakeConnection()
        repository = CollectionStateRepository(connection)

        await repository.mark_bootstrap_progress(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTCUSDT",
            timeframe="1h",
            provider_symbol="BTCUSDT",
            bootstrap_next_from=datetime(2024, 7, 22, tzinfo=UTC),
            completed_at=datetime(2026, 7, 18, tzinfo=UTC),
            last_successful_close_time=datetime(2026, 7, 18, 11, tzinfo=UTC),
        )
        await repository.mark_incremental_progress(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTCUSDT",
            timeframe="1h",
            provider_symbol="BTCUSDT",
            last_successful_close_time=datetime(2026, 7, 18, 12, tzinfo=UTC),
            updated_at=datetime(2026, 7, 18, 12, 1, tzinfo=UTC),
        )

        self.assertEqual(len(connection.statements), 2)
