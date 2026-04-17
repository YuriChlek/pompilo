from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Iterable

from trading.application.ports import MarketDataSynchronizer
from trading.application.services import TradingCycleService
from utils.db_actions import create_live_state_tables, create_tables


logger = logging.getLogger(__name__)


async def wait_until_next_run(target_minute: int = 0, target_second: int = 10) -> None:
    """Sleep until the next hourly run aligned to the configured minute and second."""
    if not 0 <= target_minute < 60 or not 0 <= target_second < 60:
        raise ValueError("target_minute та target_second мають бути в діапазоні 0-59")

    now = datetime.now()
    next_run = now.replace(microsecond=0)

    if now.minute > target_minute or (now.minute == target_minute and now.second >= target_second):
        next_run += timedelta(hours=1)

    next_run = next_run.replace(minute=target_minute, second=target_second)
    sleep_seconds = (next_run - now).total_seconds()
    logger.info("Sleeping for %.1f seconds until %s", sleep_seconds, next_run)
    await asyncio.sleep(sleep_seconds)


class TradingScheduler:
    """Coordinates schema preparation and recurring execution of trading cycles."""

    def __init__(
        self,
        trading_cycle: TradingCycleService,
        market_data_synchronizer: MarketDataSynchronizer | None = None,
    ) -> None:
        """Initialize recurring live trading orchestration dependencies."""
        self.trading_cycle = trading_cycle
        self.market_data_synchronizer = market_data_synchronizer

    async def run_forever(
        self,
        symbols: Iterable[str],
        *,
        target_minute: int = 0,
        target_second: int = 1,
        is_test: bool = False,
    ) -> None:
        """Run schema setup, market-data sync, and trading cycles forever on schedule."""
        symbol_list = list(symbols)
        await create_tables()
        await create_live_state_tables()
        await self.trading_cycle.executor.reconcile_state(symbol_list)

        while True:
            await wait_until_next_run(target_minute=target_minute, target_second=target_second)
            if self.market_data_synchronizer is not None:
                await self.market_data_synchronizer.synchronize()
            for symbol in symbol_list:
                await self.trading_cycle.run(symbol, is_test)
