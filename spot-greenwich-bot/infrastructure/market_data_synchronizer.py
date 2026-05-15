from __future__ import annotations

from collections.abc import Iterable

from utils.config import BINANCE_D1_INTERVAL, BINANCE_H4_INTERVAL, D1_TABLE_SUFFIX, H4_INCREMENTAL_SYNC_DAYS, H4_TABLE_SUFFIX

DAILY_SYNC_LOOKBACK_DAYS = 3


def _run_api(*, days: int, timeframe: str, table_suffix: str):
    from api import run_api

    return run_api(days=days, timeframe=timeframe, table_suffix=table_suffix)


class BinanceMarketDataSynchronizer:
    """Refresh local candle storage from Binance before a trading cycle."""

    async def synchronize(self, timeframes: Iterable[str] = ("d1", "h4")) -> None:
        """Run the configured Binance candle synchronization flow."""

        selected_timeframes = {str(timeframe).lower() for timeframe in timeframes}
        if "d1" in selected_timeframes:
            await _run_api(
                days=DAILY_SYNC_LOOKBACK_DAYS,
                timeframe=BINANCE_D1_INTERVAL,
                table_suffix=D1_TABLE_SUFFIX,
            )
        if "h4" in selected_timeframes:
            await _run_api(
                days=H4_INCREMENTAL_SYNC_DAYS,
                timeframe=BINANCE_H4_INTERVAL,
                table_suffix=H4_TABLE_SUFFIX,
            )
