from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import logging
from typing import Iterable

from application.initialization_service import TradingInitializationService
from application.ports import MarketDataSynchronizer
from application.trading_cycle_service import TradingCycleService
from utils.healthcheck import mark_cycle_completed, start_healthcheck_server

logger = logging.getLogger(__name__)


async def wait_until_next_daily_run(target_hour: int, target_minute: int, target_second: int) -> None:
    """Sleep until the next configured daily execution time."""

    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    next_run = now.replace(hour=target_hour, minute=target_minute, second=target_second, microsecond=0)
    if now >= next_run:
        next_run = next_run + timedelta(days=1)
    sleep_seconds = (next_run - now).total_seconds()
    logger.info("scheduler_sleep_daily seconds=%.1f next_run=%s", sleep_seconds, next_run)
    await asyncio.sleep(sleep_seconds)


async def wait_until_next_h4_run() -> datetime:
    """Sleep until shortly after the next 4-hour UTC candle close."""

    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    current_block_hour = (now.hour // 4) * 4
    next_run = now.replace(hour=current_block_hour, minute=0, second=1, microsecond=0)
    if now >= next_run:
        next_run = next_run + timedelta(hours=4)
    sleep_seconds = (next_run - now).total_seconds()
    logger.info("scheduler_sleep_h4 seconds=%.1f next_run=%s", sleep_seconds, next_run)
    await asyncio.sleep(sleep_seconds)
    return next_run


class TradingScheduler:
    """Own the recurring live execution loop for the trading bot."""

    def __init__(
        self,
        trading_cycle: TradingCycleService,
        market_data_synchronizer: MarketDataSynchronizer | None = None,
        initialization_service: TradingInitializationService | None = None,
    ) -> None:
        self.trading_cycle = trading_cycle
        self.market_data_synchronizer = market_data_synchronizer
        self.initialization_service = initialization_service

    async def run_forever(
        self,
        symbols: Iterable[str],
        *,
        target_hour: int,
        target_minute: int,
        target_second: int,
        dry_run: bool = False,
        reconcile_positions: bool = True,
    ) -> None:
        """Run the daily scheduler loop indefinitely."""

        healthcheck_task = asyncio.create_task(start_healthcheck_server())
        if self.initialization_service is not None:
            await self.initialization_service.initialize_runtime(
                symbols,
                reconcile_positions=reconcile_positions,
            )
        logger.info("scheduler_started cadence=h4 target=00:00:01,04:00:01,08:00:01,12:00:01,16:00:01,20:00:01 timezone=UTC")
        while True:
            h4_run_at = await wait_until_next_h4_run()
            logger.info("scheduled_cycle_started run_at=%s", h4_run_at)
            if self.market_data_synchronizer is not None:
                timeframes = ("d1", "h4") if h4_run_at.hour == 0 else ("h4",)
                logger.info("scheduled_cycle_sync timeframes=%s", ",".join(timeframes))
                await self.market_data_synchronizer.synchronize(timeframes=timeframes)
            results = await self.trading_cycle.run_many(symbols, dry_run=dry_run)
            pnl_summary = self.trading_cycle.build_pnl_summary(results)
            logger.info(
                "scheduled_cycle_completed symbols=%s closed_symbols=%s total_realized_pnl=%s",
                len(results),
                ",".join(pnl_summary["closed_symbols"]),
                pnl_summary["total_realized_pnl"],
            )
            mark_cycle_completed()
            if healthcheck_task.done():
                healthcheck_task.result()


__all__ = ["TradingScheduler", "wait_until_next_daily_run", "wait_until_next_h4_run"]
