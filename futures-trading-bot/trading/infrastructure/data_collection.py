from __future__ import annotations

from api import run_api


class BinanceMarketDataSynchronizer:
    """Infrastructure adapter that refreshes candle data from Binance."""

    async def synchronize(self) -> None:
        """Run the default candle synchronization flow for configured symbols."""
        await run_api()
