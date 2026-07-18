from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolAvailabilityStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbolAvailability
from market_data_service.persistence.repositories.provider_symbol_availability_repository import (
    ProviderSymbolAvailabilityRepository,
)


class FakeExecuteResult:
    def __init__(self, rowcount: int, rows: list[dict[str, object]]) -> None:
        self.rowcount = rowcount
        self.rows = rows

    def mappings(self) -> FakeMappingsResult:
        return FakeMappingsResult(self.rows)


class FakeMappingsResult:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def one_or_none(self) -> dict[str, object] | None:
        if not self.rows:
            return None
        return self.rows[0]


class FakeConnection:
    def __init__(self, rows: list[dict[str, object]], rowcount: int = 1) -> None:
        self.rows = rows
        self.rowcount = rowcount
        self.statements: list[object] = []

    async def execute(self, statement: object) -> FakeExecuteResult:
        self.statements.append(statement)
        return FakeExecuteResult(self.rowcount, self.rows)


class ProviderSymbolAvailabilityRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_returns_none_when_no_record(self) -> None:
        connection = FakeConnection(rows=[])
        repository = ProviderSymbolAvailabilityRepository(connection)

        result = await repository.get(MarketDataSource.BINANCE_SPOT, "ETH/USDT")

        self.assertIsNone(result)
        self.assertEqual(len(connection.statements), 1)

    async def test_get_returns_mapped_availability(self) -> None:
        row = {
            "source": "BINANCE_SPOT",
            "requested_symbol": "ETH/USDT",
            "provider_symbol": "ETHUSDT",
            "status": "SUPPORTED",
            "first_seen_at": datetime(2026, 7, 14, 8, tzinfo=UTC),
            "last_checked_at": datetime(2026, 7, 14, 10, tzinfo=UTC),
            "next_check_at": datetime(2026, 7, 15, 10, tzinfo=UTC),
            "failure_reason": None,
            "metadata_json": {"test": True},
        }
        connection = FakeConnection(rows=[row])
        repository = ProviderSymbolAvailabilityRepository(connection)

        result = await repository.get(MarketDataSource.BINANCE_SPOT, "ETH/USDT")

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(result.requested_symbol, "ETH/USDT")
        self.assertEqual(result.provider_symbol, "ETHUSDT")
        self.assertEqual(result.status, ProviderSymbolAvailabilityStatus.SUPPORTED)
        self.assertEqual(result.first_seen_at, datetime(2026, 7, 14, 8, tzinfo=UTC))
        self.assertEqual(result.last_checked_at, datetime(2026, 7, 14, 10, tzinfo=UTC))
        self.assertEqual(result.next_check_at, datetime(2026, 7, 15, 10, tzinfo=UTC))
        self.assertIsNone(result.failure_reason)
        self.assertEqual(result.metadata, {"test": True})

    async def test_upsert_executes_statement(self) -> None:
        connection = FakeConnection(rows=[], rowcount=1)
        repository = ProviderSymbolAvailabilityRepository(connection)

        availability = ProviderSymbolAvailability(
            source=MarketDataSource.BINANCE_SPOT,
            requested_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            status=ProviderSymbolAvailabilityStatus.SUPPORTED,
            next_check_at=datetime(2026, 7, 15, 10, tzinfo=UTC),
            metadata={"test": True},
        )

        await repository.upsert(availability)

        self.assertEqual(len(connection.statements), 1)
