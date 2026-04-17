import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from tests.support import install_common_test_stubs


install_common_test_stubs()
import types


from trading.application.scheduler import TradingScheduler


class LiveStateSchedulerTests(unittest.TestCase):
    def test_scheduler_prepares_live_state_and_runs_reconciliation_before_loop(self):
        trading_cycle = types.SimpleNamespace(
            run=AsyncMock(return_value=None),
            executor=types.SimpleNamespace(reconcile_state=AsyncMock(return_value=None)),
        )
        scheduler = TradingScheduler(trading_cycle=trading_cycle, market_data_synchronizer=None)

        async def _stop(*args, **kwargs):
            raise RuntimeError("stop")

        with patch("trading.application.scheduler.create_tables", AsyncMock()) as create_tables_mock, patch(
            "trading.application.scheduler.create_live_state_tables", AsyncMock()
        ) as create_live_state_tables_mock, patch(
            "trading.application.scheduler.wait_until_next_run", AsyncMock(side_effect=_stop)
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(scheduler.run_forever(["SOLUSDT"]))

        create_tables_mock.assert_awaited_once()
        create_live_state_tables_mock.assert_awaited_once()
        trading_cycle.executor.reconcile_state.assert_awaited_once_with(["SOLUSDT"])
        trading_cycle.run.assert_not_awaited()
