from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.application.services.backfill_command_service import BackfillCommand, BackfillCommandService
from market_data_service.domain.enums import MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol


class BackfillCommandServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_backfill_request_is_split_into_idempotent_chunks(self) -> None:
        requester = FakeBackfillRequester()
        service = BackfillCommandService(
            symbol_registry=FakeSymbolRegistry(),
            backfill_requester=requester,
        )
        command = BackfillCommand(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol="ethusdt",
            timeframe="1h",
            from_time=datetime(2026, 7, 14, 0, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 5, tzinfo=UTC),
            batch_size_candles=2,
            max_concurrency=1,
        )

        first_result = await service.request_backfill(command)
        second_result = await service.request_backfill(command)

        self.assertEqual(first_result.chunk_count, 3)
        self.assertEqual(first_result.requested_count, 3)
        self.assertEqual(first_result.skipped_duplicate_count, 0)
        self.assertEqual(second_result.chunk_count, 3)
        self.assertEqual(second_result.requested_count, 0)
        self.assertEqual(second_result.skipped_duplicate_count, 3)
        self.assertEqual(len(requester.events_by_key), 3)
        self.assertEqual(
            [(event.requested_from.hour, event.requested_to.hour) for event in requester.events_by_key.values()],
            [(0, 2), (2, 4), (4, 5)],
        )

    async def test_backfill_rejects_invalid_range(self) -> None:
        service = BackfillCommandService(
            symbol_registry=FakeSymbolRegistry(),
            backfill_requester=FakeBackfillRequester(),
        )
        command = BackfillCommand(
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol="ETHUSDT",
            timeframe="1h",
            from_time=datetime(2026, 7, 14, 5, tzinfo=UTC),
            to_time=datetime(2026, 7, 14, 5, tzinfo=UTC),
            batch_size_candles=2,
            max_concurrency=1,
        )

        with self.assertRaisesRegex(ValueError, "earlier"):
            await service.request_backfill(command)


class FakeSymbolRegistry:
    async def get_provider_symbol(self, source: MarketDataSource, provider_symbol: str) -> ProviderSymbol:
        return ProviderSymbol(
            source=source,
            canonical_symbol="ETH/USDT",
            provider_symbol=provider_symbol,
            status=ProviderSymbolStatus.TRADING,
            supported_timeframes=("1h", "4h", "1d"),
            max_backfill_days=1095,
            metadata={},
        )


class FakeBackfillRequester:
    def __init__(self) -> None:
        self.events_by_key = {}

    async def request_backfill(self, event) -> bool:
        if event.idempotency_key in self.events_by_key:
            return False
        self.events_by_key[event.idempotency_key] = event
        return True
