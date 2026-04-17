import unittest
from unittest.mock import AsyncMock, patch

from infrastructure.db import ensure_candle_tables


class CandleStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_candle_tables_creates_schema_and_symbol_tables(self):
        conn = AsyncMock()
        with patch("infrastructure.db.create_connection", AsyncMock(return_value=conn)):
            await ensure_candle_tables(["SOLUSDT", "XRPUSDT"])

        executed_sql = [call.args[0] for call in conn.execute.await_args_list]
        self.assertEqual(len(executed_sql), 3)
        self.assertIn("CREATE SCHEMA _candles_trading_data", executed_sql[0])
        self.assertIn("_candles_trading_data.solusdt_p_candles", executed_sql[1])
        self.assertIn("_candles_trading_data.xrpusdt_p_candles", executed_sql[2])
        conn.close.assert_awaited_once()
