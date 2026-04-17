from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from indicators.trend_bot import TrendBot
from trading.domain.signals import generate_strategy_signal

from .models import BacktestConfig


class StrategyBacktestAdapter:
    """Adapt live indicator and signal logic for historical candle replay."""

    def __init__(self, config: BacktestConfig):
        """Prepare an adapter that reuses the current live signal logic in backtests."""
        self.config = config
        self.bot = TrendBot(data_fetcher=None, strategy_config=config.strategy_config)

    async def build_signal(self, symbol: str, candles_df) -> Tuple[Optional[Dict[str, Any]], Optional[Any], Any]:
        """Calculate indicators and return a trading signal for the visible history window."""
        if len(candles_df) < self.config.min_candles:
            return None, None, candles_df

        prepared_h1 = self.bot._prepare_h1_indicators(candles_df.copy())
        prepared_h4 = self.bot._prepare_h4_indicators(candles_df.copy())
        prepared_d1 = self.bot._prepare_d1_indicators(candles_df.copy())
        trend_result = self.bot._assemble_trend_result(prepared_h1, prepared_h4, prepared_d1)
        indicators_history = prepared_h1.tail(self.config.indicator_history_period)
        signal = await generate_strategy_signal(
            symbol,
            trend_result,
            indicators_history,
            True,
            strategy_config=self.config.strategy_config,
        )
        return signal, trend_result, prepared_h1
