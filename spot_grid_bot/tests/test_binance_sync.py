import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from infrastructure.binance_api import BinanceCandle, _build_candle_records, fetch_and_store
from infrastructure.binance_market_data_synchronizer import BinanceMarketDataSynchronizer


class _AcquireContext:
    def __init__(self, conn) -> None:
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakePool:
    def __init__(self, conn) -> None:
        self.conn = conn
        self.closed = False

    def acquire(self):
        return _AcquireContext(self.conn)

    async def close(self):
        self.closed = True


class _FakeBinanceAPI:
    def __init__(self, candles):
        self.candles = candles
        self.calls = []
        self.closed = False

    def fetch(self, symbol: str, start_time: int, end_time: int, interval: str):
        self.calls.append((symbol, start_time, end_time, interval))
        return list(self.candles)

    def close(self):
        self.closed = True


class BinanceSyncTests(unittest.IsolatedAsyncioTestCase):
    def test_build_candle_records_uses_shared_cvd_and_candle_id_semantics(self):
        candles = [
            BinanceCandle(
                symbol="SOLUSDT",
                open_time=datetime(2026, 1, 1, 0, 0, 0),
                close_time=datetime(2026, 1, 1, 1, 0, 0),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=10.2,
            ),
            BinanceCandle(
                symbol="SOLUSDT",
                open_time=datetime(2026, 1, 1, 1, 0, 0),
                close_time=datetime(2026, 1, 1, 2, 0, 0),
                open=101.0,
                high=103.0,
                low=100.0,
                close=100.0,
                volume=5.4,
            ),
        ]

        records = _build_candle_records(candles)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][7], 10)
        self.assertEqual(records[1][7], 5)
        self.assertEqual(records[0][9], "2026-01-01 00:00:00")
        self.assertEqual(records[1][9], "2026-01-01 01:00:00")

    async def test_fetch_and_store_writes_symbol_candles_into_timeframe_specific_table(self):
        conn = AsyncMock()
        pool = _FakePool(conn)
        candles = [
            BinanceCandle(
                symbol="SOLUSDT",
                open_time=datetime(2026, 1, 1, 0, 0, 0),
                close_time=datetime(2026, 1, 1, 1, 0, 0),
                open=100.0,
                high=102.0,
                low=99.0,
                close=101.0,
                volume=10.2,
            )
        ]
        api = _FakeBinanceAPI(candles)

        with patch("infrastructure.binance_api.insert_candles", AsyncMock()) as insert_mock, patch(
            "infrastructure.binance_api.asyncio.sleep",
            AsyncMock(),
        ):
            await fetch_and_store("SOLUSDT", days=1, pool=pool, api=api)

        insert_mock.assert_awaited_once()
        self.assertEqual(insert_mock.await_args.args[1], "_candles_trading_data")
        self.assertEqual(insert_mock.await_args.args[2], "solusdt_1h")
        self.assertEqual(insert_mock.await_args.args[3], candles)
        self.assertEqual(len(api.calls), 1)
        self.assertFalse(pool.closed)
        self.assertFalse(api.closed)

    async def test_fetch_and_store_routes_4h_candles_into_4h_table(self):
        conn = AsyncMock()
        pool = _FakePool(conn)
        candles = [
            BinanceCandle(
                symbol="SOLUSDT",
                open_time=datetime(2026, 1, 1, 0, 0, 0),
                close_time=datetime(2026, 1, 1, 4, 0, 0),
                open=100.0,
                high=104.0,
                low=99.0,
                close=103.0,
                volume=10.2,
            )
        ]
        api = _FakeBinanceAPI(candles)

        with patch("infrastructure.binance_api.insert_candles", AsyncMock()) as insert_mock, patch(
            "infrastructure.binance_api.asyncio.sleep",
            AsyncMock(),
        ):
            await fetch_and_store("SOLUSDT", timeframe="4h", days=1, pool=pool, api=api)

        insert_mock.assert_awaited_once()
        self.assertEqual(insert_mock.await_args.args[2], "solusdt_4h")
        self.assertEqual(insert_mock.await_args.args[3], candles)
        self.assertEqual(api.calls[0][3], "4h")

    async def test_fetch_and_store_rejects_unsupported_timeframe(self):
        conn = AsyncMock()
        pool = _FakePool(conn)
        api = _FakeBinanceAPI([])

        with self.assertRaises(ValueError):
            await fetch_and_store("SOLUSDT", timeframe="15m", days=1, pool=pool, api=api)

    async def test_market_data_synchronizer_delegates_to_binance_sync_runner(self):
        synchronizer = BinanceMarketDataSynchronizer(symbols=("SOLUSDT", "ETHUSDT"), timeframe="1h", days=7)
        with patch("infrastructure.binance_market_data_synchronizer.run_binance_candle_sync", AsyncMock()) as run_sync:
            await synchronizer.synchronize()

        run_sync.assert_awaited_once_with(("SOLUSDT", "ETHUSDT"), timeframe="1h", days=7)

    async def test_market_data_synchronizer_default_flow_syncs_1h_then_configured_higher_timeframe(self):
        synchronizer = BinanceMarketDataSynchronizer(
            symbols=("SOLUSDT", "ETHUSDT"),
            timeframe=None,
            higher_timeframe="4h",
            days=7,
        )
        with patch("infrastructure.binance_market_data_synchronizer.run_binance_candle_sync", AsyncMock()) as run_sync:
            await synchronizer.synchronize()

        self.assertEqual(run_sync.await_count, 2)
        first_call = run_sync.await_args_list[0]
        second_call = run_sync.await_args_list[1]
        self.assertEqual(first_call.args, (("SOLUSDT", "ETHUSDT"),))
        self.assertEqual(first_call.kwargs, {"timeframe": "1h", "days": 7})
        self.assertEqual(second_call.args, (("SOLUSDT", "ETHUSDT"),))
        self.assertEqual(second_call.kwargs, {"timeframe": "4h", "days": 7})
