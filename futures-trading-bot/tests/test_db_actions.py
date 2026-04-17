import asyncio
import unittest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

from tests.support import install_common_test_stubs


install_common_test_stubs()

from utils import db_actions


class DbActionsTests(unittest.TestCase):
    def setUp(self):
        db_actions._ensured_live_state_schemas.clear()
        db_actions._ensured_candle_tables.clear()

    def test_create_live_state_tables_ensures_schema_once_per_process(self):
        conn = AsyncMock()

        @asynccontextmanager
        async def _fake_connection():
            yield conn

        with patch("utils.db_actions._acquire_shared_connection", _fake_connection):
            asyncio.run(db_actions.create_live_state_tables())
            asyncio.run(db_actions.create_live_state_tables())

        self.assertEqual(conn.execute.await_count, 4)

    def test_sync_live_positions_uses_batch_upsert(self):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 0")
        conn.executemany = AsyncMock()
        positions = [
            {
                "positionIdx": 1,
                "direction": "Buy",
                "size": "1.5",
                "avgPrice": "100",
                "takeProfit": "110",
                "stopLoss": "95",
            },
            {
                "positionIdx": 2,
                "direction": "Sell",
                "size": "2",
                "avgPrice": "101",
                "takeProfit": "90",
                "stopLoss": "105",
            },
        ]

        @asynccontextmanager
        async def _fake_connection():
            yield conn

        with patch("utils.db_actions.create_live_state_tables", new=AsyncMock()), patch(
            "utils.db_actions._acquire_shared_connection", _fake_connection
        ):
            count = asyncio.run(db_actions.sync_live_positions("SOLUSDT", positions))

        self.assertEqual(count, 2)
        conn.executemany.assert_awaited_once()
        self.assertEqual(len(conn.executemany.await_args.args[1]), 2)
        self.assertEqual(conn.execute.await_count, 1)

    def test_sync_live_orders_uses_batch_upsert(self):
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="DELETE 0")
        conn.executemany = AsyncMock()
        orders = [
            {
                "orderId": "one",
                "side": "Buy",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "100",
                "qty": "1",
                "createdTime": "123",
            },
            {
                "orderId": "two",
                "side": "Sell",
                "orderType": "Limit",
                "orderStatus": "New",
                "price": "101",
                "qty": "2",
                "createdTime": "124",
            },
        ]

        @asynccontextmanager
        async def _fake_connection():
            yield conn

        with patch("utils.db_actions.create_live_state_tables", new=AsyncMock()), patch(
            "utils.db_actions._acquire_shared_connection", _fake_connection
        ):
            count = asyncio.run(db_actions.sync_live_orders("SOLUSDT", orders))

        self.assertEqual(count, 2)
        conn.executemany.assert_awaited_once()
        self.assertEqual(len(conn.executemany.await_args.args[1]), 2)
        self.assertEqual(conn.execute.await_count, 1)
