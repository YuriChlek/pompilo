from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Iterable

from application.health import RuntimeHealthTracker
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
        live_price_monitor=None,
        health_tracker: RuntimeHealthTracker | None = None,
    ) -> None:
        """Store the trading cycle and optional market-data synchronizer."""
        self.trading_cycle = trading_cycle
        self.market_data_synchronizer = market_data_synchronizer
        self.live_price_monitor = live_price_monitor
        self.health_tracker = health_tracker
        self._run_lock = asyncio.Lock()
        if self.live_price_monitor is not None:
            self.live_price_monitor.on_deviation = self.handle_live_price_deviation

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
        if self.health_tracker is not None:
            self.health_tracker.set_tracked_symbols(symbol_list)
        await self.trading_cycle.initialize(symbol_list)
        live_monitor_task = None
        if self.live_price_monitor is not None:
            live_monitor_task = asyncio.create_task(self.live_price_monitor.run_forever(symbol_list))

        iteration = 0
        try:
            while True:
                await wait_until_next_run(target_minute=target_minute, target_second=target_second)
                iteration += 1
                await self._run_cycle(symbol_list, iteration=iteration, reason="scheduled")
        finally:
            if live_monitor_task is not None:
                live_monitor_task.cancel()
                try:
                    await live_monitor_task
                except asyncio.CancelledError:
                    pass

    async def handle_live_price_deviation(self, event) -> None:
        """Run an off-cycle emergency trading pass for one symbol after a large price move."""
        await self._run_cycle([event.symbol], iteration=None, reason="live_price_deviation")

    async def _run_cycle(self, symbols: list[str], *, iteration: int | None, reason: str) -> None:
        async with self._run_lock:
            logger.info("scheduler_cycle_started iteration=%s symbols=%s reason=%s", iteration, len(symbols), reason)
            if self.health_tracker is not None:
                self.health_tracker.record_cycle_started()
            try:
                if self.market_data_synchronizer is not None and reason == "scheduled":
                    logger.info("market_data_sync_started iteration=%s", iteration)
                    try:
                        await self.market_data_synchronizer.synchronize()
                    except Exception:
                        logger.exception(
                            "market_data_sync_failed iteration=%s -- continuing_to_trading_cycle",
                            iteration,
                        )
                    else:
                        logger.info("market_data_sync_finished iteration=%s", iteration)

                try:
                    await self.trading_cycle.run_many(symbols)
                except Exception:
                    logger.exception(
                        "scheduler_cycle_trading_failed iteration=%s reason=%s -- continuing_to_next_cycle",
                        iteration,
                        reason,
                    )
            finally:
                if self.health_tracker is not None:
                    self.health_tracker.record_cycle_completed()
                logger.info("scheduler_cycle_finished iteration=%s symbols=%s", iteration, len(symbols))
