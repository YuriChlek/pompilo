import unittest
from dataclasses import replace
from unittest.mock import patch

try:
    from indicators.trend_bot import TrendBot
    from trading.domain.strategy_config import (
        BreakoutTrendStrategyConfig,
        DEFAULT_STRATEGY_CONFIG,
        StrategyConfig,
    )
except ModuleNotFoundError:  # pragma: no cover - depends on local test env
    TrendBot = None
    BreakoutTrendStrategyConfig = None
    DEFAULT_STRATEGY_CONFIG = None
    StrategyConfig = None


class TrendBotConfigWiringTests(unittest.TestCase):
    @unittest.skipIf(TrendBot is None, "indicator dependencies are not installed in this test environment")
    def test_market_data_provider_uses_configured_breakout_lookback_for_history_padding(self):
        from trading.infrastructure.market_data import IndicatorMarketDataProvider

        custom_strategy_config = replace(
            DEFAULT_STRATEGY_CONFIG,
            breakout=BreakoutTrendStrategyConfig(lookback_candles=42),
        )

        provider = IndicatorMarketDataProvider(strategy_config=custom_strategy_config)

        with patch("trading.infrastructure.market_data.get_of_data", return_value=("trend", "history")) as get_of_data_mock:
            trend, history = provider.get_market_context("SOLUSDT", False)

        self.assertEqual((trend, history), ("trend", "history"))
        get_of_data_mock.assert_called_once_with(
            "SOLUSDT",
            False,
            strategy_config=custom_strategy_config,
            period=47,
        )

    @unittest.skipIf(TrendBot is None, "indicator dependencies are not installed in this test environment")
    def test_trend_bot_uses_configured_analysis_candle_limit_for_context_loading(self):
        bot = TrendBot()

        with patch("indicators.trend_bot.ANALYSIS_CANDLE_LIMIT", 1800):
            with patch.object(bot, "_fetch_symbol_data", return_value=None) as fetch_mock:
                with patch.object(bot, "calculate_trend_and_history_from_dataframe", return_value=("trend", "history")):
                    trend, history = bot.calculate_trend_and_history("SOLUSDT", period=5)

        self.assertEqual((trend, history), ("trend", "history"))
        fetch_mock.assert_called_once_with("SOLUSDT", limit=1800)

    @unittest.skipIf(TrendBot is None, "indicator dependencies are not installed in this test environment")
    def test_trend_bot_preserves_history_padding_when_period_exceeds_analysis_limit(self):
        bot = TrendBot()

        with patch("indicators.trend_bot.ANALYSIS_CANDLE_LIMIT", 1500):
            with patch.object(bot, "_fetch_symbol_data", return_value=None) as fetch_mock:
                with patch.object(bot, "calculate_trend_and_history_from_dataframe", return_value=("trend", "history")):
                    bot.calculate_trend_and_history("SOLUSDT", period=2000)

        fetch_mock.assert_called_once_with("SOLUSDT", limit=2100)
