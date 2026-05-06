from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

from application.runtime_commands import RuntimeCommandService


class RuntimeCommandServiceTests(unittest.TestCase):
    def test_analyze_uses_new_cycle_and_initialization_services(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Init:
            async def initialize_runtime(self, symbols) -> None:
                calls.append(("init", tuple(symbols)))

        class _Cycle:
            async def run(self, symbol: str, dry_run: bool = False) -> None:
                calls.append(("run", (symbol, dry_run)))

        with patch("application.runtime_commands._build_initialization_service", return_value=_Init()):
            with patch("application.runtime_commands._build_live_trading_cycle", return_value=_Cycle()):
                asyncio.run(RuntimeCommandService().analyze(symbol="ETHUSDT", dry_run=True))

        self.assertEqual(calls, [("init", ("ETHUSDT",)), ("run", ("ETHUSDT", True))])

    def test_live_uses_new_scheduler_builder(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Scheduler:
            async def run_forever(self, symbols, **kwargs) -> None:
                calls.append(("live", (tuple(symbols), kwargs["dry_run"])))

        with patch("application.runtime_commands._build_live_trading_scheduler", return_value=_Scheduler()):
            asyncio.run(RuntimeCommandService().live(dry_run=True))

        self.assertEqual(calls, [("live", (("ETHUSDT", "SUIUSDT", "TAOUSDT", "SOLUSDT", "BTCUSDT", "XRPUSDT", "LTCUSDT"), True))])

    def test_init_db_and_migrate_use_new_initialization_service(self) -> None:
        calls: list[str] = []

        class _Init:
            async def create_tables(self) -> None:
                calls.append("create_tables")

            async def run_migrations(self) -> None:
                calls.append("run_migrations")

        with patch("application.runtime_commands._build_initialization_service", return_value=_Init()):
            asyncio.run(RuntimeCommandService().init_db())
            asyncio.run(RuntimeCommandService().migrate())

        self.assertEqual(calls, ["create_tables", "run_migrations"])

    def test_sync_handlers_use_run_api_helper(self) -> None:
        calls: list[int] = []

        async def _fake_run_api(*, days: int) -> None:
            calls.append(days)

        with patch("application.runtime_commands._run_api", _fake_run_api):
            asyncio.run(RuntimeCommandService().sync(days=42))
            asyncio.run(RuntimeCommandService().sync_3y())

        self.assertEqual(calls[0], 42)
        self.assertGreater(calls[1], 1000)
