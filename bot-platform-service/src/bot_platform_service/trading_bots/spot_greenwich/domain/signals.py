from __future__ import annotations

from decimal import Decimal

from bot_platform_service.trading_bots.spot_greenwich.domain.models import (
    GreenwichCandle,
    GreenwichSignalConfig,
    GreenwichSignalSnapshot,
    GreenwichSignalType,
    GreenwichSpotSignal,
)


def build_greenwich_signal_snapshot(
    candles: tuple[GreenwichCandle, ...],
    *,
    config: GreenwichSignalConfig = GreenwichSignalConfig(),
) -> GreenwichSignalSnapshot:
    """Return the latest Greenwich indicator snapshot for one candle history."""

    if len(candles) < config.length + 2:
        raise ValueError("not enough history for Greenwich signal generation")

    closes = tuple(candle.close for candle in candles)
    highs = tuple(candle.high for candle in candles)
    lows = tuple(candle.low for candle in candles)
    basis = _basis(closes, config.basis_type, config.length)
    true_range = _true_range(highs, lows, closes)
    atr = _rma(true_range, config.length)

    upper1 = _band(basis, atr, config.multiplier_1, add=True)
    upper2 = _band(basis, atr, config.multiplier_2, add=True)
    upper3 = _band(basis, atr, config.multiplier_3, add=True)
    lower1 = _band(basis, atr, config.multiplier_1, add=False)
    lower2 = _band(basis, atr, config.multiplier_2, add=False)
    lower3 = _band(basis, atr, config.multiplier_3, add=False)
    last_index = len(candles) - 1
    previous_index = last_index - 1
    if lower3[previous_index] is None or lower3[last_index] is None or upper2[previous_index] is None or upper2[last_index] is None:
        raise ValueError("not enough complete Greenwich band values")

    buy_signal = lows[previous_index] <= lower3[previous_index] and lows[last_index] > lower3[last_index]
    sell_signal = closes[previous_index] >= upper2[previous_index] and closes[last_index] < upper2[last_index]
    return GreenwichSignalSnapshot(
        basis=_required_decimal(basis[last_index]),
        upper1=_required_decimal(upper1[last_index]),
        upper2=_required_decimal(upper2[last_index]),
        upper3=_required_decimal(upper3[last_index]),
        lower1=_required_decimal(lower1[last_index]),
        lower2=_required_decimal(lower2[last_index]),
        lower3=_required_decimal(lower3[last_index]),
        buy_signal=buy_signal,
        sell_signal=sell_signal,
        signal_price=candles[last_index].close,
        signal_high=candles[last_index].high,
        close_time=candles[last_index].timestamp,
    )


def build_take_profit_signal(
    symbol: str,
    snapshot: GreenwichSignalSnapshot,
    *,
    timeframe: str = "1d",
) -> GreenwichSpotSignal:
    return GreenwichSpotSignal(
        symbol=symbol.upper(),
        signal_type=GreenwichSignalType.SELL,
        signal_price=snapshot.upper1,
        close_time=snapshot.close_time,
        reason="greenwich_take_profit_upper1",
        timeframe=timeframe,
        candle_id=f"{snapshot.close_time}:upper1_take_profit",
    )


def generate_spot_signal(
    symbol: str,
    candles: tuple[GreenwichCandle, ...],
    *,
    timeframe: str = "1d",
    config: GreenwichSignalConfig = GreenwichSignalConfig(),
) -> GreenwichSpotSignal:
    """Build the Greenwich signal for one symbol from platform-native candles."""

    snapshot = build_greenwich_signal_snapshot(candles, config=config)
    if snapshot.buy_signal:
        return GreenwichSpotSignal(
            symbol=symbol.upper(),
            signal_type=GreenwichSignalType.BUY,
            signal_price=snapshot.signal_price,
            close_time=snapshot.close_time,
            reason="greenwich_buy_recovery",
            timeframe=timeframe,
            candle_id=snapshot.close_time,
        )
    if snapshot.sell_signal:
        return GreenwichSpotSignal(
            symbol=symbol.upper(),
            signal_type=GreenwichSignalType.SELL,
            signal_price=snapshot.signal_price,
            close_time=snapshot.close_time,
            reason="greenwich_sell_fade",
            timeframe=timeframe,
            candle_id=snapshot.close_time,
        )
    return GreenwichSpotSignal(
        symbol=symbol.upper(),
        signal_type=GreenwichSignalType.HOLD,
        signal_price=snapshot.signal_price,
        close_time=snapshot.close_time,
        reason="no_greenwich_signal",
        timeframe=timeframe,
        candle_id=snapshot.close_time,
    )


def resolve_atr_size_multiplier(
    candles: tuple[GreenwichCandle, ...],
    *,
    config: GreenwichSignalConfig = GreenwichSignalConfig(),
) -> Decimal:
    if not config.atr_position_sizing_enabled:
        return Decimal("1")
    required_length = max(config.length + 2, config.atr_position_sizing_median_window + 1)
    if len(candles) < required_length:
        return Decimal("1")
    true_range = _true_range(
        tuple(candle.high for candle in candles),
        tuple(candle.low for candle in candles),
        tuple(candle.close for candle in candles),
    )
    atr = tuple(value for value in _rma(true_range, config.length) if value is not None)
    if len(atr) < config.atr_position_sizing_median_window:
        return Decimal("1")
    current_atr = atr[-1]
    median_atr = _median(atr[-config.atr_position_sizing_median_window :])
    if current_atr <= 0 or median_atr <= 0:
        return Decimal("1")
    multiplier = median_atr / current_atr
    if multiplier < config.atr_position_sizing_min_multiplier:
        return config.atr_position_sizing_min_multiplier
    if multiplier > config.atr_position_sizing_max_multiplier:
        return config.atr_position_sizing_max_multiplier
    return multiplier


def _basis(values: tuple[Decimal, ...], basis_type: str, length: int) -> tuple[Decimal | None, ...]:
    normalized = basis_type.upper()
    if normalized == "EMA":
        return _ema(values, length)
    if normalized == "SMA":
        return _sma(values, length)
    if normalized == "RMA":
        return _rma(values, length)
    return _wma(values, length)


def _true_range(
    highs: tuple[Decimal, ...],
    lows: tuple[Decimal, ...],
    closes: tuple[Decimal, ...],
) -> tuple[Decimal, ...]:
    ranges: list[Decimal] = []
    for index, high in enumerate(highs):
        low = lows[index]
        if index == 0:
            ranges.append(high - low)
            continue
        previous_close = closes[index - 1]
        ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return tuple(ranges)


def _wma(values: tuple[Decimal, ...], length: int) -> tuple[Decimal | None, ...]:
    weights = tuple(Decimal(index) for index in range(1, length + 1))
    weight_sum = sum(weights)
    result: list[Decimal | None] = []
    for index in range(len(values)):
        if index + 1 < length:
            result.append(None)
            continue
        window = values[index + 1 - length : index + 1]
        result.append(sum(item * weight for item, weight in zip(window, weights, strict=True)) / weight_sum)
    return tuple(result)


def _sma(values: tuple[Decimal, ...], length: int) -> tuple[Decimal | None, ...]:
    result: list[Decimal | None] = []
    divisor = Decimal(length)
    for index in range(len(values)):
        if index + 1 < length:
            result.append(None)
            continue
        result.append(sum(values[index + 1 - length : index + 1]) / divisor)
    return tuple(result)


def _ema(values: tuple[Decimal, ...], length: int) -> tuple[Decimal | None, ...]:
    alpha = Decimal("2") / Decimal(length + 1)
    result: list[Decimal | None] = []
    current: Decimal | None = None
    for index, value in enumerate(values):
        if current is None:
            current = value
        else:
            current = (value * alpha) + (current * (Decimal("1") - alpha))
        result.append(current if index + 1 >= length else None)
    return tuple(result)


def _rma(values: tuple[Decimal, ...], length: int) -> tuple[Decimal | None, ...]:
    alpha = Decimal("1") / Decimal(length)
    result: list[Decimal | None] = []
    current: Decimal | None = None
    for index, value in enumerate(values):
        if current is None:
            current = value
        else:
            current = (value * alpha) + (current * (Decimal("1") - alpha))
        result.append(current if index + 1 >= length else None)
    return tuple(result)


def _band(
    basis: tuple[Decimal | None, ...],
    atr: tuple[Decimal | None, ...],
    multiplier: Decimal,
    *,
    add: bool,
) -> tuple[Decimal | None, ...]:
    values: list[Decimal | None] = []
    for basis_value, atr_value in zip(basis, atr, strict=True):
        if basis_value is None or atr_value is None:
            values.append(None)
            continue
        offset = multiplier * atr_value
        values.append(basis_value + offset if add else basis_value - offset)
    return tuple(values)


def _median(values: tuple[Decimal, ...]) -> Decimal:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal("2")


def _required_decimal(value: Decimal | None) -> Decimal:
    if value is None:
        raise ValueError("missing Greenwich value")
    return value

