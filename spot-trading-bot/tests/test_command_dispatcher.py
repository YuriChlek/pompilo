from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest

from application.command_dispatcher import dispatch_command


class CommandDispatcherTests(unittest.TestCase):
    def test_dispatch_routes_to_expected_handler(self) -> None:
        calls: list[tuple[str, object]] = []

        class _Handlers:
            async def sync(self, *, days: int) -> None:
                calls.append(("sync", days))

            async def sync_3y(self) -> None:
                calls.append(("sync-3y", None))

            async def analyze(self, *, symbol=None, dry_run: bool = False) -> None:
                calls.append(("analyze", (symbol, dry_run)))

            async def init_db(self) -> None:
                calls.append(("init-db", None))

            async def migrate(self) -> None:
                calls.append(("migrate", None))

            async def live(self, *, dry_run: bool = False) -> None:
                calls.append(("live", dry_run))

        asyncio.run(dispatch_command(SimpleNamespace(command="sync", period=123, dry_run=False, symbol=None), _Handlers()))
        asyncio.run(dispatch_command(SimpleNamespace(command="analyze", period=None, dry_run=True, symbol="ETHUSDT"), _Handlers()))
        asyncio.run(dispatch_command(SimpleNamespace(command="migrate", period=None, dry_run=False, symbol=None), _Handlers()))
        asyncio.run(dispatch_command(SimpleNamespace(command=None, period=None, dry_run=True, symbol=None), _Handlers()))

        self.assertEqual(
            calls,
            [
                ("sync", 123),
                ("analyze", ("ETHUSDT", True)),
                ("migrate", None),
                ("live", True),
            ],
        )
