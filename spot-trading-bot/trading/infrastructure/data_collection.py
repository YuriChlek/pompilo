from __future__ import annotations

from api import run_api


class BinanceMarketDataSynchronizer:
    async def synchronize(self) -> None:
        await run_api()
