from __future__ import annotations

from api import run_api


class BinanceMarketDataSynchronizer:
    """Refresh local candle storage from Binance before a trading cycle."""

    async def synchronize(self) -> None:
        """Run the configured Binance candle synchronization flow."""

        await run_api()
