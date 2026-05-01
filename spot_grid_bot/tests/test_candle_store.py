import unittest
from unittest.mock import AsyncMock, patch

from infrastructure.db import ensure_candle_tables, resolve_candle_table_name


class CandleStoreTests(unittest.IsolatedAsyncioTestCase):
    def test_resolve_candle_table_name_uses_only_new_timeframe_suffixes(self):
        self.assertEqual(resolve_candle_table_name("SOLUSDT", timeframe="1h"), "solusdt_1h")
        self.assertEqual(resolve_candle_table_name("SOLUSDT", timeframe="4h"), "solusdt_4h")
        self.assertNotIn("_p_candles", resolve_candle_table_name("SOLUSDT", timeframe="1h"))

    async def test_ensure_candle_tables_creates_schema_and_symbol_tables(self):
        conn = AsyncMock()
        with patch("infrastructure.db.create_connection", AsyncMock(return_value=conn)):
            await ensure_candle_tables(["SOLUSDT", "XRPUSDT"])

        executed_sql = [call.args[0] for call in conn.execute.await_args_list]
        self.assertEqual(len(executed_sql), 5)
        self.assertIn("CREATE SCHEMA _candles_trading_data", executed_sql[0])
        self.assertIn("_candles_trading_data.solusdt_1h", executed_sql[1])
        self.assertIn("_candles_trading_data.solusdt_4h", executed_sql[2])
        self.assertIn("_candles_trading_data.xrpusdt_1h", executed_sql[3])
        self.assertIn("_candles_trading_data.xrpusdt_4h", executed_sql[4])
        conn.close.assert_awaited_once()
