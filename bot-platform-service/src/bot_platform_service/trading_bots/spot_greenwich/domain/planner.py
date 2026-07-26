from __future__ import annotations

from decimal import Decimal

from bot_platform_service.trading_bots.spot_greenwich.domain.execution import decide_spot_execution
from bot_platform_service.trading_bots.spot_greenwich.domain.models import (
    GreenwichCandle,
    GreenwichConfig,
    GreenwichMultiTimeframePlan,
    GreenwichMultiTimeframeSignal,
    GreenwichPositionState,
    GreenwichSignalType,
    GreenwichSpotSignal,
    GreenwichTradingPlan,
)
from bot_platform_service.trading_bots.spot_greenwich.domain.signals import (
    build_greenwich_signal_snapshot,
    build_take_profit_signal,
    generate_spot_signal,
    resolve_atr_size_multiplier,
)


class GreenwichSpotPlanner:
    """Planner facade for one Greenwich spot timeframe."""

    def __init__(self, config: GreenwichConfig | None = None) -> None:
        self.config = config or GreenwichConfig(symbols=("ETHUSDT",), primary_timeframe="1d", supporting_timeframes=("4h",))

    def plan(
        self,
        *,
        symbol: str,
        candles: tuple[GreenwichCandle, ...],
        position_state: GreenwichPositionState,
        available_quote_balance: Decimal,
        timeframe: str = "1d",
    ) -> GreenwichTradingPlan:
        """Build both the signal and the decision for one symbol."""

        signal = self._build_signal(symbol.upper(), candles, position_state, timeframe=timeframe)
        atr_size_multiplier = resolve_atr_size_multiplier(candles, config=self.config.signal)
        decision = decide_spot_execution(
            signal,
            position_state,
            available_quote_balance,
            config=self.config.execution,
            atr_size_multiplier=atr_size_multiplier,
        )
        return GreenwichTradingPlan(
            signal=signal,
            decision=decision,
            diagnostics={
                "planner": "spot_greenwich_decimal_bands",
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "candle_count": len(candles),
                "atr_size_multiplier": str(atr_size_multiplier),
            },
        )

    def _build_signal(
        self,
        symbol: str,
        candles: tuple[GreenwichCandle, ...],
        position_state: GreenwichPositionState,
        *,
        timeframe: str,
    ) -> GreenwichSpotSignal:
        base_signal = generate_spot_signal(symbol, candles, timeframe=timeframe, config=self.config.signal)
        filtered_signal = _apply_signal_filters(symbol, base_signal, candles, config=self.config)
        if filtered_signal.signal_type is GreenwichSignalType.SELL:
            return filtered_signal
        if filtered_signal.signal_type is not base_signal.signal_type or filtered_signal.reason != base_signal.reason:
            return filtered_signal
        if position_state.has_position and not position_state.first_take_profit_done:
            snapshot = build_greenwich_signal_snapshot(candles, config=self.config.signal)
            if snapshot.signal_high >= snapshot.upper1:
                return build_take_profit_signal(symbol, snapshot, timeframe=timeframe)
        return filtered_signal


class MultiTimeframeSpotPlanner:
    """Apply the D1 regime filter to 4H Greenwich entry signals."""

    def __init__(
        self,
        config: GreenwichConfig | None = None,
        *,
        d1_regime_filter_enabled: bool = True,
    ) -> None:
        self.config = config or GreenwichConfig(symbols=("ETHUSDT",), primary_timeframe="4h", supporting_timeframes=("1d",))
        self.d1_regime_filter_enabled = d1_regime_filter_enabled

    def plan(
        self,
        *,
        symbol: str,
        candles: dict[str, tuple[GreenwichCandle, ...]],
        position_state: GreenwichPositionState,
        available_quote_balance: Decimal,
    ) -> GreenwichMultiTimeframePlan:
        """Build the 4H decision after applying the D1 regime filter."""

        signal = self.build_signal(symbol=symbol, candles=candles, position_state=position_state)
        atr_size_multiplier = resolve_atr_size_multiplier(candles["4h"], config=self.config.signal)
        decision = decide_spot_execution(
            signal.resolved,
            position_state,
            available_quote_balance,
            config=self.config.execution,
            atr_size_multiplier=atr_size_multiplier,
        )
        return GreenwichMultiTimeframePlan(
            signal=signal,
            decision=decision,
            diagnostics={
                "planner": "spot_greenwich_multi_timeframe",
                "symbol": symbol.upper(),
                "primary_timeframe": "4h",
                "supporting_timeframes": ("1d",),
                "d1_regime_blocked": signal.d1_regime_blocked,
                "h4_candle_count": len(candles["4h"]),
                "d1_candle_count": len(candles["1d"]),
                "atr_size_multiplier": str(atr_size_multiplier),
            },
        )

    def build_signal(
        self,
        *,
        symbol: str,
        candles: dict[str, tuple[GreenwichCandle, ...]],
        position_state: GreenwichPositionState | None = None,
    ) -> GreenwichMultiTimeframeSignal:
        """Return the raw 4H signal and resolved multi-timeframe signal."""

        resolved_position_state = position_state or GreenwichPositionState(symbol.upper(), Decimal("0"), Decimal("0"), Decimal("0"))
        d1_regime_blocked = self._is_d1_buy_blocked(candles["1d"])
        h4_signal = self._build_h4_signal(symbol.upper(), candles["4h"], resolved_position_state)
        resolved_signal = self._resolve(d1_regime_blocked, h4_signal)
        return GreenwichMultiTimeframeSignal(
            symbol=symbol.upper(),
            d1_regime_blocked=d1_regime_blocked,
            h4=h4_signal,
            resolved=resolved_signal,
        )

    def _is_d1_buy_blocked(self, d1_candles: tuple[GreenwichCandle, ...]) -> bool:
        if not self.d1_regime_filter_enabled or len(d1_candles) < 201:
            return False

        close = tuple(candle.close for candle in d1_candles)
        fast = _wma(close, 50)
        slow = _wma(close, 200)
        if fast[-2] is None or fast[-1] is None or slow[-2] is None or slow[-1] is None:
            return False
        death_cross = fast[-2] >= slow[-2] and fast[-1] < slow[-1]
        if not death_cross:
            return False
        adx = _calculate_adx(d1_candles)
        return bool(adx[-1] is not None and adx[-1] >= Decimal("30"))

    def _build_h4_signal(
        self,
        symbol: str,
        h4_candles: tuple[GreenwichCandle, ...],
        position_state: GreenwichPositionState,
    ) -> GreenwichSpotSignal:
        planner = GreenwichSpotPlanner(self.config)
        return planner._build_signal(symbol, h4_candles, position_state, timeframe="4h")

    def _resolve(self, d1_regime_blocked: bool, h4: GreenwichSpotSignal) -> GreenwichSpotSignal:
        if d1_regime_blocked and h4.signal_type is GreenwichSignalType.BUY:
            return GreenwichSpotSignal(
                h4.symbol,
                GreenwichSignalType.HOLD,
                h4.signal_price,
                h4.close_time,
                "d1_regime_blocks_h4_buy",
                h4.timeframe,
                h4.candle_id,
            )
        return h4


def _hold_signal_from(signal: GreenwichSpotSignal, reason: str) -> GreenwichSpotSignal:
    return GreenwichSpotSignal(
        signal.symbol,
        GreenwichSignalType.HOLD,
        signal.signal_price,
        signal.close_time,
        reason,
        signal.timeframe,
        signal.candle_id,
    )


def _resolve_previous_buy_signal(
    symbol: str,
    candles: tuple[GreenwichCandle, ...],
    timeframe: str,
    *,
    config: GreenwichConfig,
) -> GreenwichSpotSignal | None:
    if len(candles) < config.signal.length + 3:
        return None
    try:
        previous_signal = generate_spot_signal(symbol, candles[:-1], timeframe=timeframe, config=config.signal)
    except ValueError:
        return None
    if previous_signal.signal_type is not GreenwichSignalType.BUY:
        return None
    return previous_signal


def _resolve_confirmation_buy_signal(
    symbol: str,
    signal: GreenwichSpotSignal,
    candles: tuple[GreenwichCandle, ...],
    *,
    config: GreenwichConfig,
) -> GreenwichSpotSignal:
    if not config.signal.confirmation_candle_enabled or signal.signal_type is GreenwichSignalType.SELL:
        return signal
    previous_buy_signal = _resolve_previous_buy_signal(symbol, candles, signal.timeframe, config=config)
    if signal.signal_type is GreenwichSignalType.BUY and previous_buy_signal is None:
        return _hold_signal_from(signal, "buy_waiting_confirmation")
    if previous_buy_signal is None:
        return signal
    latest_snapshot = build_greenwich_signal_snapshot(candles, config=config.signal)
    latest_close = candles[-1].close
    if latest_close <= latest_snapshot.lower3:
        return GreenwichSpotSignal(
            symbol.upper(),
            GreenwichSignalType.HOLD,
            latest_close,
            latest_snapshot.close_time,
            "buy_confirmation_failed",
            signal.timeframe,
            latest_snapshot.close_time,
        )
    return GreenwichSpotSignal(
        symbol.upper(),
        GreenwichSignalType.BUY,
        latest_close,
        latest_snapshot.close_time,
        "greenwich_buy_confirmation",
        signal.timeframe,
        latest_snapshot.close_time,
    )


def _passes_anti_crash_buy_block(
    signal: GreenwichSpotSignal,
    candles: tuple[GreenwichCandle, ...],
    *,
    config: GreenwichConfig,
) -> bool:
    if not config.signal.anti_crash_buy_block_enabled or signal.signal_type is not GreenwichSignalType.BUY:
        return True
    required_length = config.signal.anti_crash_lookback_candles + 1
    if len(candles) < required_length:
        return True
    start_close = candles[-required_length].close
    current_close = candles[-1].close
    if start_close <= 0:
        return True
    drop_ratio = (start_close - current_close) / start_close
    return drop_ratio <= config.signal.anti_crash_max_drop_ratio


def _apply_anti_crash_buy_block(
    signal: GreenwichSpotSignal,
    candles: tuple[GreenwichCandle, ...],
    *,
    config: GreenwichConfig,
) -> GreenwichSpotSignal:
    if _passes_anti_crash_buy_block(signal, candles, config=config):
        return signal
    return _hold_signal_from(signal, "buy_anti_crash_blocked")


def _apply_signal_filters(
    symbol: str,
    base_signal: GreenwichSpotSignal,
    candles: tuple[GreenwichCandle, ...],
    *,
    config: GreenwichConfig,
) -> GreenwichSpotSignal:
    resolved_signal = _resolve_confirmation_buy_signal(symbol, base_signal, candles, config=config)
    return _apply_anti_crash_buy_block(resolved_signal, candles, config=config)


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


def _calculate_adx(candles: tuple[GreenwichCandle, ...], length: int = 14) -> tuple[Decimal | None, ...]:
    plus_dm: list[Decimal] = [Decimal("0")]
    minus_dm: list[Decimal] = [Decimal("0")]
    true_range: list[Decimal] = [candles[0].high - candles[0].low]
    for index in range(1, len(candles)):
        current = candles[index]
        previous = candles[index - 1]
        up_move = current.high - previous.high
        down_move = previous.low - current.low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else Decimal("0"))
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else Decimal("0"))
        true_range.append(max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close)))

    atr = _rma(tuple(true_range), length)
    plus_di = _ratio_series(_rma(tuple(plus_dm), length), atr)
    minus_di = _ratio_series(_rma(tuple(minus_dm), length), atr)
    dx: list[Decimal] = []
    for plus_value, minus_value in zip(plus_di, minus_di, strict=True):
        if plus_value is None or minus_value is None or plus_value + minus_value == 0:
            dx.append(Decimal("0"))
            continue
        dx.append((abs(plus_value - minus_value) / (plus_value + minus_value)) * Decimal("100"))
    return _rma(tuple(dx), length)


def _ratio_series(
    numerator: tuple[Decimal | None, ...],
    denominator: tuple[Decimal | None, ...],
) -> tuple[Decimal | None, ...]:
    values: list[Decimal | None] = []
    for numerator_value, denominator_value in zip(numerator, denominator, strict=True):
        if numerator_value is None or denominator_value is None or denominator_value == 0:
            values.append(None)
            continue
        values.append((Decimal("100") * numerator_value) / denominator_value)
    return tuple(values)

