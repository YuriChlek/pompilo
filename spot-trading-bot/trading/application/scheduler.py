from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Iterable

from trading.application.ports import MarketDataSynchronizer
from trading.application.services import TradingCycleService
from utils.db_actions import create_tables


async def wait_until_next_daily_run(target_hour: int, target_minute: int, target_second: int) -> None:
    now = datetime.now()
    next_run = now.replace(hour=target_hour, minute=target_minute, second=target_second, microsecond=0)
    if now >= next_run:
        next_run = next_run + timedelta(days=1)
    sleep_seconds = (next_run - now).total_seconds()
    print(f"🕒 Sleeping for {sleep_seconds:.1f} seconds until {next_run}")
    await asyncio.sleep(sleep_seconds)


class TradingScheduler:
    def __init__(self, trading_cycle: TradingCycleService, market_data_synchronizer: MarketDataSynchronizer | None = None) -> None:
        self.trading_cycle = trading_cycle
        self.market_data_synchronizer = market_data_synchronizer

    async def run_forever(
        self,
        symbols: Iterable[str],
        *,
        target_hour: int,
        target_minute: int,
        target_second: int,
        dry_run: bool = False,
    ) -> None:
        await create_tables()
        print("✅ Spot scheduler started")
        print(
            f"📅 Daily execution target: {target_hour:02d}:{target_minute:02d}:{target_second:02d}"
        )
        while True:
            await wait_until_next_daily_run(target_hour, target_minute, target_second)
            print("🔄 Starting scheduled spot cycle")
            if self.market_data_synchronizer is not None:
                print("📥 Synchronizing D1 candles from Binance")
                await self.market_data_synchronizer.synchronize()
            for symbol in symbols:
                print(f"🧠 Processing symbol {symbol}")
                await self.trading_cycle.run(symbol, dry_run=dry_run)
            print("✅ Scheduled spot cycle completed")
