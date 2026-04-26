from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from application.scheduler import TradingScheduler


class TradingSchedulerTests(unittest.TestCase):
    def test_run_forever_initializes_then_syncs_then_runs_cycle(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Cycle:
            async def run(self, symbol: str, dry_run: bool = False):
                calls.append(("run", (symbol, dry_run)))
                raise StopAsyncIteration

        class _Sync:
            async def synchronize(self) -> None:
                calls.append(("sync", None))

        class _Init:
            async def initialize_runtime(self, symbols) -> None:
                calls.append(("init", tuple(symbols)))

        scheduler = TradingScheduler(
            trading_cycle=_Cycle(),
            market_data_synchronizer=_Sync(),
            initialization_service=_Init(),
        )

        async def _no_wait(*args, **kwargs) -> None:
            calls.append(("wait", None))

        with patch("application.scheduler.wait_until_next_daily_run", _no_wait):
            with self.assertRaises(StopAsyncIteration):
                asyncio.run(
                    scheduler.run_forever(
                        ["ETHUSDT"],
                        target_hour=0,
                        target_minute=5,
                        target_second=0,
                        dry_run=True,
                    )
                )

        self.assertEqual(
            calls,
            [
                ("init", ("ETHUSDT",)),
                ("wait", None),
                ("sync", None),
                ("run", ("ETHUSDT", True)),
            ],
        )
