from __future__ import annotations

import asyncio
import unittest

from application.initialization_service import TradingInitializationService


class TradingInitializationServiceTests(unittest.TestCase):
    def test_initialize_runtime_runs_table_setup_before_optional_reconciliation(self) -> None:
        calls: list[tuple[str, str | None]] = []

        async def _create_tables() -> None:
            calls.append(("tables", None))

        async def _run_migrations() -> None:
            calls.append(("migrations", None))

        class _Executor:
            async def get_position_state(self, symbol: str):
                calls.append(("reconcile", symbol))
                return None

        service = TradingInitializationService(
            table_initializer=_create_tables,
            migration_runner=_run_migrations,
            executor=_Executor(),
        )

        asyncio.run(service.initialize_runtime(["ETHUSDT", "BTCUSDT"], reconcile_positions=True))

        self.assertEqual(
            calls,
            [("tables", None), ("reconcile", "ETHUSDT"), ("reconcile", "BTCUSDT")],
        )

    def test_run_migrations_uses_migration_runner_only(self) -> None:
        calls: list[str] = []

        async def _create_tables() -> None:
            calls.append("tables")

        async def _run_migrations() -> None:
            calls.append("migrations")

        service = TradingInitializationService(
            table_initializer=_create_tables,
            migration_runner=_run_migrations,
        )

        asyncio.run(service.run_migrations())

        self.assertEqual(calls, ["migrations"])
