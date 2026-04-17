from __future__ import annotations

import math

from domain.models import Candle, IndicatorSnapshot
from domain.strategy_config import StrategyConfig


def ema(values: list[float], length: int) -> list[float]:
    """Return an exponentially weighted moving average series."""
    if not values:
        return []
    multiplier = 2.0 / (length + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append((value - result[-1]) * multiplier + result[-1])
    return result


def true_ranges(candles: list[Candle]) -> list[float]:
    """Return true-range values for a candle series."""
    if not candles:
        return []
    ranges = [candles[0].high - candles[0].low]
    for previous, current in zip(candles, candles[1:]):
        ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return ranges


def rolling_mean(values: list[float], length: int) -> list[float]:
    """Return a simple rolling mean series for the requested window length."""
    if not values:
        return []
    result: list[float] = []
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= length:
            running -= values[index - length]
        denom = min(index + 1, length)
        result.append(running / denom)
    return result


def atr(candles: list[Candle], length: int) -> list[float]:
    """Return an ATR-like rolling mean of true ranges."""
    return rolling_mean(true_ranges(candles), length)


def realized_volatility(candles: list[Candle], length: int) -> list[float]:
    """Return realized log-return volatility for each candle."""
    if len(candles) < 2:
        return [0.0] * len(candles)
    returns = [0.0]
    for previous, current in zip(candles, candles[1:]):
        returns.append(math.log(current.close / previous.close) if previous.close > 0 else 0.0)
    vols: list[float] = []
    for index in range(len(returns)):
        window = returns[max(0, index - length + 1) : index + 1]
        mean = sum(window) / len(window)
        variance = sum((item - mean) ** 2 for item in window) / len(window)
        vols.append(math.sqrt(variance))
    return vols


def compute_snapshot(candles: list[Candle], config: StrategyConfig) -> IndicatorSnapshot:
    """Build the indicator snapshot used by regime, risk, and strategy planning."""
    closes = [candle.close for candle in candles]
    ema20 = ema(closes, config.regime.ema_fast_length)
    ema50 = ema(closes, config.regime.ema_mid_length)
    ema200 = ema(closes, config.regime.ema_slow_length)
    atr_values = atr(candles, config.regime.atr_length)
    rv_values = realized_volatility(candles, config.regime.realized_vol_length)
    lookback = min(config.regime.regime_lookback, len(candles))
    recent = candles[-lookback:]
    width = max(candle.high for candle in recent) - min(candle.low for candle in recent)
    latest_close = closes[-1]
    slope_base = ema50[-min(6, len(ema50))]
    slope = (ema50[-1] - slope_base) / slope_base if slope_base else 0.0
    directional_move = abs(latest_close - closes[-lookback]) / max(atr_values[-1], 1e-9) if lookback > 1 else 0.0
    last_candle = candles[-1]
    latest_atr = atr_values[-1]
    avg_atr = sum(atr_values[-lookback:]) / lookback
    abnormal_candle = (last_candle.high - last_candle.low) > latest_atr * config.regime.abnormal_candle_atr_multiplier
    atr_spike = latest_atr > avg_atr * config.regime.atr_spike_multiplier
    return IndicatorSnapshot(
        ema20=ema20[-1],
        ema50=ema50[-1],
        ema200=ema200[-1],
        atr14=latest_atr,
        realized_volatility=rv_values[-1],
        ema50_slope=slope,
        range_width=width,
        price_vs_ema50=(latest_close - ema50[-1]) / ema50[-1] if ema50[-1] else 0.0,
        directional_move=directional_move,
        abnormal_candle=abnormal_candle,
        atr_spike=atr_spike,
    )
