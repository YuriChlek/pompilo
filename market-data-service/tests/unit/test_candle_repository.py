from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import MarketDataSource
from market_data_service.persistence.repositories.candle_repository import CandleRepository


class FakeExecuteResult:
    def __init__(self, row: tuple[datetime] | None) -> None:
        self.row = row

    def fetchone(self) -> tuple[datetime] | None:
        return self.row


class FakeConnection:
    def __init__(self, row: tuple[datetime] | None) -> None:
        self.row = row
        self.statements: list[object] = []

    async def execute(self, statement):
        self.statements.append(statement)
        return FakeExecuteResult(self.row)


class CandleRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_latest_closed_candle_time_returns_datetime(self) -> None:
        expected_time = datetime(2026, 7, 18, 12, tzinfo=UTC)
        connection = FakeConnection(row=(expected_time,))
        repository = CandleRepository(connection)

        result = await repository.get_latest_closed_candle_time(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTC/USDT",
            timeframe="1h",
        )

        self.assertEqual(result, expected_time)
        self.assertEqual(len(connection.statements), 1)

    async def test_get_latest_closed_candle_time_returns_none_if_empty(self) -> None:
        connection = FakeConnection(row=None)
        repository = CandleRepository(connection)

        result = await repository.get_latest_closed_candle_time(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="BTC/USDT",
            timeframe="1h",
        )

        self.assertIsNone(result)
        self.assertEqual(len(connection.statements), 1)
