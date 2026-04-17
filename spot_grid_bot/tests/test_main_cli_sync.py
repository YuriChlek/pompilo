import argparse
import unittest
from unittest.mock import AsyncMock, patch

import main


class MainCliSyncTests(unittest.IsolatedAsyncioTestCase):
    def test_positive_period_rejects_non_positive_values(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            main._positive_period("0")

        with self.assertRaises(argparse.ArgumentTypeError):
            main._positive_period("-3")

        self.assertEqual(main._positive_period("7"), 7)

    async def test_run_sync_ensures_tables_before_binance_sync(self):
        call_order: list[str] = []

        async def _ensure_tables(symbols):
            call_order.append("ensure")

        async def _run_sync(symbols, timeframe: str, days: int):
            call_order.append("sync")

        with patch("main.ensure_candle_tables", AsyncMock(side_effect=_ensure_tables)), patch(
            "main.run_binance_candle_sync",
            AsyncMock(side_effect=_run_sync),
        ):
            await main._run_sync(30, "1h")

        self.assertEqual(call_order, ["ensure", "sync"])

    async def test_start_dispatches_sync_command(self):
        with patch("main._build_parser") as build_parser, patch("main._run_sync", AsyncMock()) as run_sync:
            build_parser.return_value.parse_args.return_value = argparse.Namespace(command="sync", period=14, timeframe="4h")
            await main.start()

        run_sync.assert_awaited_once_with(14, "4h")
