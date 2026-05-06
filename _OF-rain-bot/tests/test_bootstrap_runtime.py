from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from trading.application.bootstrap import build_trading_runtime
from trading.application.runner import CanonicalTradingRuntime, run_trading_application
from trading.infrastructure.execution_service import BybitExecutionService


class BootstrapRuntimeTests(unittest.TestCase):
    def test_build_trading_runtime_returns_canonical_runtime(self) -> None:
        runtime = build_trading_runtime(dry_run=True)

        self.assertIsInstance(runtime, CanonicalTradingRuntime)
        self.assertTrue(runtime.dry_run)
        self.assertTrue(runtime.bot.dry_run)
        self.assertIsInstance(runtime.bot.executor, BybitExecutionService)
        self.assertIs(runtime.bot.trading_service.runtime, runtime.bot)


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_trading_application_uses_bootstrap_runtime(self) -> None:
        fake_runtime = type(
            "_Runtime",
            (),
            {
                "start": AsyncMock(),
                "close": AsyncMock(),
            },
        )()

        with patch("trading.application.bootstrap.build_trading_runtime", return_value=fake_runtime) as build_runtime:
            await run_trading_application(dry_run=True)

        build_runtime.assert_called_once_with(dry_run=True)
        fake_runtime.start.assert_awaited_once()
        fake_runtime.close.assert_awaited_once()
