from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Iterable

from application.ports import MarketDataSynchronizer
from application.trading_cycle_service import SpotTradingCycleService

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
    logger.info("sleeping_until_next_run seconds=%.1f target=%s", sleep_seconds, next_run)
    await asyncio.sleep(sleep_seconds)


class TradingScheduler:
    """Coordinate recurring execution of trading cycles for configured symbols."""

    def __init__(
        self,
        trading_cycle: SpotTradingCycleService,
        market_data_synchronizer: MarketDataSynchronizer | None = None,
    ) -> None:
        """Store the trading cycle and optional market-data synchronizer."""
        self.trading_cycle = trading_cycle
        self.market_data_synchronizer = market_data_synchronizer

    async def run_forever(
        self,
        symbols: Iterable[str],
        *,
        target_minute: int = 0,
        target_second: int = 1,
    ) -> None:
        """Initialize execution state once and then run trading cycles on schedule."""
        symbol_list = list(symbols)
        logger.info(
            "scheduler_started symbols=%s target_minute=%s target_second=%s",
            ",".join(symbol_list),
            target_minute,
            target_second,
        )
        await self.trading_cycle.initialize(symbol_list)

        iteration = 0
        while True:
            await wait_until_next_run(target_minute=target_minute, target_second=target_second)
            iteration += 1
            logger.info("scheduler_cycle_started iteration=%s symbols=%s", iteration, len(symbol_list))
            if self.market_data_synchronizer is not None:
                logger.info("market_data_sync_started iteration=%s", iteration)
                await self.market_data_synchronizer.synchronize()
                logger.info("market_data_sync_finished iteration=%s", iteration)
            await self.trading_cycle.run_many(symbol_list)
            logger.info("scheduler_cycle_finished iteration=%s symbols=%s", iteration, len(symbol_list))
