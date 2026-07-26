from __future__ import annotations

from datetime import UTC, datetime
import unittest

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.infrastructure.providers.routing_candle_provider import RoutingCandleProvider


class MockCandleProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[ProviderSymbol, str, datetime, datetime]] = []
        self.return_value = ["candle1", "candle2"]

    async def fetch_closed_candles(
        self,
        provider_symbol: ProviderSymbol,
        *,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[str]:
        self.calls.append((provider_symbol, timeframe, from_time, to_time))
        return self.return_value


class RoutingCandleProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_routes_to_correct_provider(self) -> None:
        binance_mock = MockCandleProvider()
        bybit_mock = MockCandleProvider()
        
        provider = RoutingCandleProvider({
            MarketDataSource.BINANCE_SPOT: binance_mock,
            MarketDataSource.BYBIT_SPOT: bybit_mock,
        })
        
        symbol_binance = ProviderSymbol(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            status=ProviderSymbolStatus.TRADING,
            supported_timeframes=("1h",),
        )
        
        now = datetime.now(UTC)
        result = await provider.fetch_closed_candles(
            symbol_binance,
            timeframe="1h",
            from_time=now,
            to_time=now,
        )
        
        self.assertEqual(result, ["candle1", "candle2"])
        self.assertEqual(len(binance_mock.calls), 1)
        self.assertEqual(len(bybit_mock.calls), 0)
        self.assertEqual(binance_mock.calls[0][0], symbol_binance)

    async def test_raises_value_error_if_source_not_registered(self) -> None:
        provider = RoutingCandleProvider({})
        symbol = ProviderSymbol(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            status=ProviderSymbolStatus.TRADING,
            supported_timeframes=("1h",),
        )
        
        now = datetime.now(UTC)
        with self.assertRaises(ValueError) as context:
            await provider.fetch_closed_candles(
                symbol,
                timeframe="1h",
                from_time=now,
                to_time=now,
            )
        self.assertIn("No adapter registered for source", str(context.exception))
