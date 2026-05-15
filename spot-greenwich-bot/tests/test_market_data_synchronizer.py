from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from infrastructure.market_data_synchronizer import BinanceMarketDataSynchronizer
from utils.config import H4_INCREMENTAL_SYNC_DAYS


class BinanceMarketDataSynchronizerTests(unittest.TestCase):
    def test_daily_synchronization_uses_short_overlap_window(self) -> None:
        calls: list[int] = []

        async def _fake_run_api(*, days: int, timeframe: str, table_suffix: str) -> None:
            calls.append((days, timeframe, table_suffix))

        with patch("infrastructure.market_data_synchronizer._run_api", _fake_run_api):
            asyncio.run(BinanceMarketDataSynchronizer().synchronize())

        self.assertEqual(calls, [(3, "1d", "_1d"), (H4_INCREMENTAL_SYNC_DAYS, "4h", "_4h")])

    def test_can_synchronize_h4_only(self) -> None:
        calls: list[tuple[int, str, str]] = []

        async def _fake_run_api(*, days: int, timeframe: str, table_suffix: str) -> None:
            calls.append((days, timeframe, table_suffix))

        with patch("infrastructure.market_data_synchronizer._run_api", _fake_run_api):
            asyncio.run(BinanceMarketDataSynchronizer().synchronize(timeframes=("h4",)))

        self.assertEqual(calls, [(H4_INCREMENTAL_SYNC_DAYS, "4h", "_4h")])
