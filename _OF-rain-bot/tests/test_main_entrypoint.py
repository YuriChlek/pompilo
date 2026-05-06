from __future__ import annotations

import unittest
from argparse import Namespace
from unittest.mock import AsyncMock, patch

import main


class MainEntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_start_uses_canonical_runner_and_closes_db_pool(self) -> None:
        with (
            patch.object(main, "_build_parser") as build_parser,
            patch.object(main, "run_trading_application", new=AsyncMock()) as run_application,
            patch.object(main, "close_db_pool", new=AsyncMock()) as close_db_pool,
        ):
            build_parser.return_value.parse_args.return_value = Namespace(dry_run=True)

            await main.start()

        run_application.assert_awaited_once_with(dry_run=True)
        close_db_pool.assert_awaited_once()

    async def test_start_closes_db_pool_when_runner_fails(self) -> None:
        with (
            patch.object(main, "_build_parser") as build_parser,
            patch.object(main, "run_trading_application", new=AsyncMock(side_effect=RuntimeError("boom"))),
            patch.object(main, "close_db_pool", new=AsyncMock()) as close_db_pool,
        ):
            build_parser.return_value.parse_args.return_value = Namespace(dry_run=False)

            with self.assertRaisesRegex(RuntimeError, "boom"):
                await main.start()

        close_db_pool.assert_awaited_once()
