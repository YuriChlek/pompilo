from __future__ import annotations

import asyncio
from datetime import datetime
import unittest
from unittest.mock import patch

from application.scheduler import TradingScheduler, wait_until_next_h4_run


class TradingSchedulerTests(unittest.TestCase):
    def test_run_forever_initializes_then_syncs_then_runs_cycle(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Cycle:
            async def run_many(self, symbols, dry_run: bool = False):
                calls.append(("run_many", (tuple(symbols), dry_run)))
                raise StopAsyncIteration

        class _Sync:
            async def synchronize(self, timeframes=("d1", "h4")) -> None:
                calls.append(("sync", tuple(timeframes)))

        class _Init:
            async def initialize_runtime(self, symbols, *, reconcile_positions: bool = False) -> None:
                calls.append(("init", (tuple(symbols), reconcile_positions)))

        scheduler = TradingScheduler(
            trading_cycle=_Cycle(),
            market_data_synchronizer=_Sync(),
            initialization_service=_Init(),
        )

        async def _no_wait(*args, **kwargs) -> datetime:
            calls.append(("wait", None))
            return datetime(2026, 5, 8, 0, 0, 1)

        with patch("application.scheduler.wait_until_next_h4_run", _no_wait):
            with self.assertRaises(StopAsyncIteration):
                asyncio.run(
                    scheduler.run_forever(
                        ["ETHUSDT"],
                        target_hour=0,
                        target_minute=0,
                        target_second=1,
                        dry_run=True,
                    )
                )

        self.assertEqual(
            calls,
            [
                ("init", (("ETHUSDT",), True)),
                ("wait", None),
                ("sync", ("d1", "h4")),
                ("run_many", (("ETHUSDT",), True)),
            ],
        )

    def test_run_forever_can_skip_startup_reconciliation(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Cycle:
            async def run_many(self, symbols, dry_run: bool = False):
                raise StopAsyncIteration

        class _Init:
            async def initialize_runtime(self, symbols, *, reconcile_positions: bool = False) -> None:
                calls.append(("init", reconcile_positions))

        scheduler = TradingScheduler(
            trading_cycle=_Cycle(),
            initialization_service=_Init(),
        )

        async def _no_wait(*args, **kwargs) -> datetime:
            return datetime(2026, 5, 8, 4, 0, 1)

        with patch("application.scheduler.wait_until_next_h4_run", _no_wait):
            with self.assertRaises(StopAsyncIteration):
                asyncio.run(
                    scheduler.run_forever(
                        ["ETHUSDT"],
                        target_hour=0,
                        target_minute=0,
                        target_second=1,
                        reconcile_positions=False,
                    )
                )

        self.assertEqual(calls, [("init", False)])

    def test_run_forever_syncs_h4_only_after_non_midnight_h4_close(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Cycle:
            async def run_many(self, symbols, dry_run: bool = False):
                calls.append(("run_many", tuple(symbols)))
                raise StopAsyncIteration

        class _Sync:
            async def synchronize(self, timeframes=("d1", "h4")) -> None:
                calls.append(("sync", tuple(timeframes)))

        scheduler = TradingScheduler(
            trading_cycle=_Cycle(),
            market_data_synchronizer=_Sync(),
        )

        async def _no_wait(*args, **kwargs) -> datetime:
            return datetime(2026, 5, 8, 4, 0, 1)

        with patch("application.scheduler.wait_until_next_h4_run", _no_wait):
            with self.assertRaises(StopAsyncIteration):
                asyncio.run(
                    scheduler.run_forever(
                        ["ETHUSDT"],
                        target_hour=0,
                        target_minute=0,
                        target_second=1,
                    )
                )

        self.assertEqual(calls, [("sync", ("h4",)), ("run_many", ("ETHUSDT",))])

    def test_wait_until_next_h4_run_uses_utc_now(self) -> None:
        calls: list[object] = []

        class _Datetime:
            @classmethod
            def now(cls, *, tz=None):
                calls.append(("tz", tz))
                return datetime(2026, 5, 7, 3, 59, 50, tzinfo=tz)

        async def _sleep(seconds: float) -> None:
            calls.append(("sleep", seconds))

        with patch("application.scheduler.datetime", _Datetime):
            with patch("application.scheduler.asyncio.sleep", _sleep):
                run_at = asyncio.run(wait_until_next_h4_run())

        self.assertEqual(calls[0][0], "tz")
        self.assertIsNotNone(calls[0][1])
        self.assertEqual(calls[1], ("sleep", 11.0))
        self.assertEqual(run_at, datetime(2026, 5, 7, 4, 0, 1))
