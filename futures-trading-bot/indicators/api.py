from __future__ import annotations

import logging
from typing import Tuple

import pandas as pd
from trading.domain.strategy_config import DEFAULT_STRATEGY_CONFIG, StrategyConfig

from .models import TrendResult
from .trend_bot import TrendBot


logger = logging.getLogger(__name__)


def get_trend_data(
    symbol: str,
    is_test: bool = False,
    strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> TrendResult:
    """Return the latest indicator snapshot for one symbol."""
    bot = TrendBot(strategy_config=strategy_config)
    return bot.calculate_trend_for_symbol(symbol, is_test)


def get_trend_history(
    symbol: str,
    period: int = 100,
    is_test: bool = False,
    strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> pd.DataFrame:
    """Return prepared indicator history for one symbol."""
    bot = TrendBot(strategy_config=strategy_config)
    return bot.calculate_history_for_symbol(symbol, period=period, is_test=is_test)


def get_of_data(
    symbol: str,
    is_test: bool = False,
    strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
    period: int | None = None,
) -> Tuple[TrendResult, pd.DataFrame]:
    """Return both current trend data and recent indicator history for one symbol."""
    bot = TrendBot(strategy_config=strategy_config)
    try:
        history_period = period if period is not None else max(5, strategy_config.breakout.lookback_candles + 5)
        trend_data, indicators_history = bot.calculate_trend_and_history(symbol, period=history_period, is_test=is_test)
        if is_test:
            _print_test_report(symbol, trend_data)
        return trend_data, indicators_history
    except Exception as exc:
        logger.exception("indicator_context_fetch_failed symbol=%s error=%s", symbol, exc)
        raise


def get_of_data_from_dataframe(
    symbol: str,
    raw_h1: pd.DataFrame,
    is_test: bool = False,
    strategy_config: StrategyConfig = DEFAULT_STRATEGY_CONFIG,
) -> Tuple[TrendResult, pd.DataFrame]:
    """Return current trend data and recent history from a preloaded H1 dataframe."""
    bot = TrendBot(strategy_config=strategy_config)
    try:
        trend_data, indicators_history = bot.calculate_trend_and_history_from_dataframe(raw_h1, period=5)
        if is_test:
            _print_test_report(symbol, trend_data)
        return trend_data, indicators_history
    except Exception as exc:
        logger.exception("indicator_context_from_dataframe_failed symbol=%s error=%s", symbol, exc)
        raise


def _print_test_report(symbol: str, trend_data: TrendResult) -> None:
    """Print a verbose debug report for manual inspection in test mode."""
    print("=" * 80, "\n")
    if trend_data.timestamp is not None:
        print(f"⏰ Час закриття свічки: {trend_data.timestamp}")
    print("✅ Технічний аналіз успішно завершено")
    print(f"📊 Символ: {symbol}")

    print("\n" + "=" * 40 + " ОСТАННЯ СВІЧКА " + "=" * 40)
    print(f"📅 Час: {trend_data.candle['open_time']}")
    print(f"📈 Open: {trend_data.candle['open']:.4f}")
    print(f"📊 High: {trend_data.candle['high']:.4f}")
    print(f"📉 Low: {trend_data.candle['low']:.4f}")
    print(f"💰 Close: {trend_data.candle['close']:.4f}")

    print("\n" + "=" * 40 + " SUPER TREND " + "=" * 40)
    print(f"🎯 SuperTrend H1: {trend_data.super_trend:.4f}")
    print(f"📶 Сигнал H1: {trend_data.super_trend_signal}")
    print(f"🎯 SuperTrend H4: {trend_data.super_trend_h4:.4f}")
    print(f"📶 Сигнал H4: {trend_data.super_trend_h4_signal}")
    print(f"🎯 SuperTrend D1: {trend_data.super_trend_d1:.4f}")
    print(f"📶 Сигнал D1: {trend_data.super_trend_d1_signal}")

    _print_range_block("H1", trend_data.range_analysis_h1)
    _print_range_block("H4", trend_data.range_analysis_h4)
    _print_range_block("D1", trend_data.range_analysis_d1)
    _print_fractals_block("H1", trend_data.fractal_analysis_h1)
    _print_fractals_block("H4", trend_data.fractal_analysis_h4)
    _print_fractals_block("D1", trend_data.fractal_analysis_d1)

    print("\n" + "=" * 40 + " ЗАГАЛЬНА ІНФОРМАЦІЯ " + "=" * 40)
    print(f"💰 Ціна закриття: {trend_data.indicators['close']:.4f}")
    print(f"📉 ATR: {trend_data.atr:.4f}")
    print("\n" + "=" * 80 + "\n")


def _print_range_block(label: str, range_analysis) -> None:
    """Print a formatted range-analysis section for manual debug output."""
    print("\n" + "=" * 40 + f" АНАЛІЗ ДІАПАЗОНУ {label} " + "=" * 40)
    if not range_analysis:
        print("Дані відсутні")
        return
    print(f"📊 ДІАПАЗОН {label}: {'✅ ТАК' if range_analysis.is_range else '❌ НІ'}")
    print(f"🎯 Впевненість: {range_analysis.confidence:.1f}%")
    print(f"📋 Тип: {_get_range_type_description(range_analysis.range_type)}")
    print(f"💪 Сила кластеризації: {range_analysis.cluster_strength}")
    print(f"🔮 Паттерн: {range_analysis.fractal_pattern}")
    print(f"📉 ATR: {range_analysis.atr_pct:.2f}%")
    print(
        "🔢 Аналізовано фракталів: "
        f"{range_analysis.upper_fractals_analyzed} верхніх, "
        f"{range_analysis.lower_fractals_analyzed} нижніх"
    )
    if range_analysis.price_levels:
        print("\n📊 РІВНІ ЦІН НА ОСНОВІ ОСТАННІХ ФРАКТАЛІВ:")
        for key, value in range_analysis.price_levels.items():
            label = key.replace('_', ' ').title()
            if 'range_pct' in key:
                print(f"  • {label}: {value:.2f}%")
            else:
                print(f"  • {label}: {value:.4f}")
    if range_analysis.reasons:
        print("\n📝 ПРИЧИНИ ВИЗНАЧЕННЯ ДІАПАЗОНУ:")
        for reason in range_analysis.reasons:
            print(f"  • {reason}")


def _print_fractals_block(label: str, fractal_analysis) -> None:
    """Print a formatted fractal-analysis section for manual debug output."""
    print("\n" + "=" * 40 + f" ОСТАННІ ФРАКТАЛИ {label} " + "=" * 40)
    if not fractal_analysis:
        print("Дані відсутні")
        return
    print(f"📊 Загальна кількість верхніх фракталів: {len(fractal_analysis.upper_fractals)}")
    print(f"📊 Загальна кількість нижніх фракталів: {len(fractal_analysis.lower_fractals)}")
    if fractal_analysis.last_3_upper_fractals:
        print(
            f"\n🎯 ОСТАННІ {len(fractal_analysis.last_3_upper_fractals)} ВЕРХНІХ ФРАКТАЛА ({label}):"
        )
        for i, fractal in enumerate(fractal_analysis.last_3_upper_fractals, start=1):
            newest_info = " 🆕" if fractal.get('is_newest') else ""
            position_info = f" ({fractal.get('position')})" if fractal.get('position') else ""
            print(f"  {i}. Ціна: {fractal['price']:.4f}{position_info}{newest_info}")
    if fractal_analysis.last_3_lower_fractals:
        print(
            f"\n🎯 ОСТАННІ {len(fractal_analysis.last_3_lower_fractals)} НИЖНІХ ФРАКТАЛА ({label}):"
        )
        for i, fractal in enumerate(fractal_analysis.last_3_lower_fractals, start=1):
            newest_info = " 🆕" if fractal.get('is_newest') else ""
            position_info = f" ({fractal.get('position')})" if fractal.get('position') else ""
            print(f"  {i}. Ціна: {fractal['price']:.4f}{position_info}{newest_info}")


def _get_range_type_description(range_type: str) -> str:
    """Translate range-type codes to human-readable Ukrainian labels."""
    descriptions = {
        'upper_cluster': '🎯 Кластеризація ВЕРХНІХ фракталів (формуючийся опір)',
        'lower_cluster': '🛡️ Кластеризація НИЖНІХ фракталів (формуюча підтримка)',
        'both_cluster': '📊 Кластеризація ОБОХ типів фракталів (чіткий діапазон)',
        'range': '📉 Вузький діапазон',
        'none': '🚀 Без діапазону (тренд)'
    }
    return descriptions.get(range_type, range_type)
