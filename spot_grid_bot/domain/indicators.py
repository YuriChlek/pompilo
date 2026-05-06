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


def rsi(candles: list[Candle], length: int) -> list[float]:
    """Return a simple RSI series for the provided candle closes."""
    if not candles:
        return []
    if len(candles) == 1:
        return [50.0]

    closes = [candle.close for candle in candles]
    gains = [0.0]
    losses = [0.0]
    for previous, current in zip(closes, closes[1:]):
        delta = current - previous
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))

    avg_gains = rolling_mean(gains, length)
    avg_losses = rolling_mean(losses, length)
    values: list[float] = []
    for avg_gain, avg_loss in zip(avg_gains, avg_losses):
        if avg_loss <= 0 and avg_gain <= 0:
            values.append(50.0)
            continue
        if avg_loss <= 0:
            values.append(100.0)
            continue
        rs = avg_gain / avg_loss
        values.append(100.0 - (100.0 / (1.0 + rs)))
    return values


def compute_snapshot(candles: list[Candle], config: StrategyConfig) -> IndicatorSnapshot:
    """Build the indicator snapshot used by regime, risk, and strategy planning."""
    closes = [candle.close for candle in candles]
    volumes = [candle.volume for candle in candles]
    ema20 = ema(closes, config.regime.ema_fast_length)
    ema50 = ema(closes, config.regime.ema_mid_length)
    ema200 = ema(closes, config.regime.ema_slow_length)
    atr_values = atr(candles, config.regime.atr_length)
    rv_short_values = realized_volatility(candles, config.regime.realized_vol_short_length)
    rv_values = realized_volatility(candles, config.regime.realized_vol_length)
    rsi_values = rsi(candles, config.regime.atr_length)
    volume_ma20 = rolling_mean(volumes, 20)
    lookback = min(config.regime.regime_lookback, len(candles))
    recent = candles[-lookback:]
    width = max(candle.high for candle in recent) - min(candle.low for candle in recent)
    recent_low = min(candle.low for candle in recent)
    recent_high = max(candle.high for candle in recent)
    latest_close = closes[-1]
    slope_base = ema50[-min(6, len(ema50))]
    slope = (ema50[-1] - slope_base) / slope_base if slope_base else 0.0
    raw_directional_move = latest_close - closes[-lookback] if lookback > 1 else 0.0
    directional_move = abs(raw_directional_move) / max(atr_values[-1], 1e-9) if lookback > 1 else 0.0
    directional_sign = 0.0
    if raw_directional_move > 0:
        directional_sign = 1.0
    elif raw_directional_move < 0:
        directional_sign = -1.0
    last_candle = candles[-1]
    latest_atr = atr_values[-1]
    avg_atr = sum(atr_values[-lookback:]) / lookback
    abnormal_candle = (last_candle.high - last_candle.low) > latest_atr * config.regime.abnormal_candle_atr_multiplier
    atr_spike = latest_atr > avg_atr * config.regime.atr_spike_multiplier
    range_position = 0.5
    if recent_high > recent_low:
        range_position = (latest_close - recent_low) / (recent_high - recent_low)
        range_position = max(min(range_position, 1.0), 0.0)
    volatility_regime_ratio = 1.0
    if rv_values[-1] > 0:
        volatility_regime_ratio = rv_short_values[-1] / rv_values[-1]
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
        directional_sign=directional_sign,
        abnormal_candle=abnormal_candle,
        atr_spike=atr_spike,
        range_position=range_position,
        rsi14=rsi_values[-1],
        current_volume=volumes[-1],
        volume_ma20=volume_ma20[-1],
        realized_volatility_short=rv_short_values[-1],
        volatility_regime_ratio=volatility_regime_ratio,
    )
