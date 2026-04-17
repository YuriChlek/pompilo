from __future__ import annotations

from typing import Any, Tuple

from indicators import TrendResult, get_of_data
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig


class IndicatorMarketDataProvider:
    """Infrastructure adapter that reads market context from the indicator subsystem."""

    def __init__(self, strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG) -> None:
        self.strategy_config = strategy_config

    def get_market_context(self, symbol: str, is_test: bool) -> Tuple[TrendResult, Any]:
        """Return a trend snapshot and indicator history for the requested symbol."""
        history_period = max(5, self.strategy_config.breakout.lookback_candles + 5)
        return get_of_data(symbol, is_test, strategy_config=self.strategy_config, period=history_period)
