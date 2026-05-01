import unittest
from unittest.mock import AsyncMock, patch, MagicMock
from infrastructure.db import DatabaseCandleRepository, resolve_candle_table_name, ensure_candle_tables
from infrastructure.market_data_provider import DatabaseMarketDataProvider
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG
from domain.models import InventorySnapshot, Candle

class Phase10StorageMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_candle_tables_creates_both_1h_and_4h_tables(self):
        conn = AsyncMock()
        with patch("infrastructure.db.create_connection", AsyncMock(return_value=conn)):
            await ensure_candle_tables(["ETHUSDT"])
        
        executed_sql = [call.args[0] for call in conn.execute.await_args_list]
        self.assertTrue(any("ethusdt_1h" in sql for sql in executed_sql), "1h table not found in executed SQL")
        self.assertTrue(any("ethusdt_4h" in sql for sql in executed_sql), "4h table not found in executed SQL")

    def test_database_market_data_provider_initializes_with_correct_suffixes(self):
        exchange = MagicMock()
        provider = DatabaseMarketDataProvider(DEFAULT_STRATEGY_CONFIG, exchange)
        
        self.assertEqual(provider.repository.table_suffix, "_1h")
        self.assertEqual(provider.higher_timeframe_repository.table_suffix, "_4h")

    async def test_database_market_data_provider_loads_1h_and_4h_independently(self):
        exchange = MagicMock()
        exchange.get_balances.return_value = InventorySnapshot(0.0, 1000.0, 0.0, 2000.0)
        exchange.get_open_orders.return_value = []
        exchange.get_instrument_filters.return_value = MagicMock(tick_size=0.1, qty_step=0.01, min_order_qty=0.01, min_order_amt=10.0)
        
        provider = DatabaseMarketDataProvider(DEFAULT_STRATEGY_CONFIG, exchange)
        
        fake_1h_candle = Candle(timestamp=1000, open=1900.0, high=2100.0, low=1800.0, close=2000.0, volume=10.0)
        fake_4h_candle = Candle(timestamp=1000, open=1800.0, high=2200.0, low=1700.0, close=1950.0, volume=40.0)

        with patch.object(provider.repository, "fetch_recent_candles", AsyncMock(return_value=[fake_1h_candle])) as fetch_1h, \
             patch.object(provider.higher_timeframe_repository, "fetch_recent_candles", AsyncMock(return_value=[fake_4h_candle])) as fetch_4h:
            
            context = await provider.get_market_context("ETHUSDT")
            
            fetch_1h.assert_awaited_once_with("ETHUSDT", DEFAULT_STRATEGY_CONFIG.market_data.candles_lookback)
            fetch_4h.assert_awaited_once_with("ETHUSDT", DEFAULT_STRATEGY_CONFIG.market_data.candles_lookback // 4)
            
            self.assertEqual(context.candles[0].close, 2000.0)
            self.assertEqual(context.higher_timeframe_candles[0].close, 1950.0)

if __name__ == "__main__":
    unittest.main()
