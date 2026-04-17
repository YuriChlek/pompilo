from __future__ import annotations

from infrastructure.binance_api import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TIMEFRAME,
    run_binance_candle_sync,
)
from utils.config import TRADING_SYMBOLS


class BinanceMarketDataSynchronizer:
    """Infrastructure adapter that refreshes candle data from Binance Spot."""

    def __init__(
        self,
        *,
        symbols: tuple[str, ...] | None = None,
        timeframe: str = DEFAULT_TIMEFRAME,
        days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        """Store the candle sync configuration used by scheduled refreshes."""
        self.symbols = tuple(symbols or tuple(TRADING_SYMBOLS))
        self.timeframe = timeframe
        self.days = days

    async def synchronize(self) -> None:
        """Run the default candle synchronization flow for configured symbols."""
        await run_binance_candle_sync(self.symbols, timeframe=self.timeframe, days=self.days)
