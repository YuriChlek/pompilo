from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from domain.execution import decide_spot_execution
from domain.models import ExecutionDecision, MultiTimeframeSignal, PositionState, SpotSignal
from domain.signals import build_greenwich_signal_snapshot, build_take_profit_signal, generate_spot_signal
from utils.config import (
    ANTI_CRASH_BUY_BLOCK_ENABLED,
    ANTI_CRASH_LOOKBACK_CANDLES,
    ANTI_CRASH_MAX_DROP_RATIO,
    ATR_POSITION_SIZING_ENABLED,
    ATR_POSITION_SIZING_MAX_MULTIPLIER,
    ATR_POSITION_SIZING_MEDIAN_WINDOW,
    ATR_POSITION_SIZING_MIN_MULTIPLIER,
    CONFIRMATION_CANDLE_ENABLED,
    GREENWICH_LENGTH,
)

BUY_SIGNAL = "buy"
HOLD_SIGNAL = "hold"
SELL_SIGNAL = "sell"


@dataclass(frozen=True)
class SpotTradingPlan:
    """Combined output of Greenwich signal generation and execution planning."""

    signal: SpotSignal
    decision: ExecutionDecision


class GreenwichSpotPlanner:
    """Planner facade for the current Greenwich-based spot strategy."""

    def plan(
        self,
        symbol: str,
        candles_df,
        position_state: PositionState,
        available_quote_balance,
    ) -> SpotTradingPlan:
        """Build both the signal and the execution decision for one symbol."""

        signal = self._build_signal(symbol, candles_df, position_state)
        atr_size_multiplier = _resolve_atr_size_multiplier(candles_df)
        decision = decide_spot_execution(
            signal,
            position_state,
            available_quote_balance,
            atr_size_multiplier=atr_size_multiplier,
        )
        return SpotTradingPlan(signal=signal, decision=decision)

    def _build_signal(self, symbol: str, candles_df, position_state: PositionState) -> SpotSignal:
        base_signal = generate_spot_signal(symbol, candles_df)
        filtered_signal = _apply_signal_filters(symbol, base_signal, candles_df)
        if filtered_signal.signal_type == SELL_SIGNAL:
            return filtered_signal
        if filtered_signal.signal_type != base_signal.signal_type or filtered_signal.reason != base_signal.reason:
            return filtered_signal
        if position_state.has_position and not position_state.first_take_profit_done:
            snapshot = build_greenwich_signal_snapshot(candles_df)
            if snapshot.signal_high >= snapshot.upper1:
                return build_take_profit_signal(symbol, snapshot)
        return filtered_signal


def _wma(values, length: int):
    weights = list(range(1, length + 1))
    weight_sum = sum(weights)
    return values.rolling(length).apply(lambda window: sum(item * weight for item, weight in zip(window, weights)) / weight_sum, raw=True)


def _rma(values, length: int):
    return values.ewm(alpha=1 / length, adjust=False).mean()


def _hold_signal_from(signal: SpotSignal, reason: str) -> SpotSignal:
    return SpotSignal(
        signal.symbol,
        HOLD_SIGNAL,
        signal.signal_price,
        signal.close_time,
        reason,
        signal.timeframe,
        signal.candle_id,
    )


def _calculate_adx(candles_df, length: int = 14):
    high = candles_df["high"].astype("float64")
    low = candles_df["low"].astype("float64")
    close = candles_df["close"].astype("float64")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    true_range = candles_df.assign(
        high_low=(high - low).abs(),
        high_close=(high - prev_close).abs(),
        low_close=(low - prev_close).abs(),
    )[["high_low", "high_close", "low_close"]].max(axis=1)

    atr = _rma(true_range, length)
    plus_di = 100 * _rma(plus_dm, length) / atr
    minus_di = 100 * _rma(minus_dm, length) / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    return _rma(dx, length)


def _calculate_greenwich_atr(candles_df, length: int = GREENWICH_LENGTH):
    high = candles_df["high"].astype("float64")
    low = candles_df["low"].astype("float64")
    close = candles_df["close"].astype("float64")
    prev_close = close.shift(1)
    true_range = candles_df.assign(
        high_low=(high - low).abs(),
        high_close=(high - prev_close).abs(),
        low_close=(low - prev_close).abs(),
    )[["high_low", "high_close", "low_close"]].max(axis=1)
    return _rma(true_range.astype("float64"), length)


def _clamp_atr_size_multiplier(multiplier: Decimal) -> Decimal:
    if multiplier < ATR_POSITION_SIZING_MIN_MULTIPLIER:
        return ATR_POSITION_SIZING_MIN_MULTIPLIER
    if multiplier > ATR_POSITION_SIZING_MAX_MULTIPLIER:
        return ATR_POSITION_SIZING_MAX_MULTIPLIER
    return multiplier


def _resolve_atr_size_multiplier(candles_df) -> Decimal:
    if not ATR_POSITION_SIZING_ENABLED:
        return Decimal("1")
    if not hasattr(candles_df, "__len__") or not hasattr(candles_df, "__getitem__"):
        return Decimal("1")
    try:
        required_length = max(GREENWICH_LENGTH + 2, ATR_POSITION_SIZING_MEDIAN_WINDOW + 1)
        if len(candles_df) < required_length:
            return Decimal("1")
        atr = _calculate_greenwich_atr(candles_df)
        current_atr = Decimal(str(atr.iloc[-1]))
        median_atr = Decimal(str(atr.iloc[-ATR_POSITION_SIZING_MEDIAN_WINDOW:].median()))
    except Exception:
        return Decimal("1")
    if current_atr <= 0 or median_atr <= 0:
        return Decimal("1")
    return _clamp_atr_size_multiplier(median_atr / current_atr)


def _supports_candle_frame(candles_df) -> bool:
    return hasattr(candles_df, "__len__") and hasattr(candles_df, "__getitem__")


def _resolve_previous_buy_signal(symbol: str, candles_df, timeframe: str) -> SpotSignal | None:
    if not _supports_candle_frame(candles_df) or len(candles_df) < GREENWICH_LENGTH + 3:
        return None
    try:
        previous_signal = generate_spot_signal(symbol, candles_df.iloc[:-1], timeframe=timeframe)
    except Exception:
        return None
    if previous_signal.signal_type != BUY_SIGNAL:
        return None
    return previous_signal


def _resolve_confirmation_buy_signal(symbol: str, signal: SpotSignal, candles_df) -> SpotSignal:
    if not CONFIRMATION_CANDLE_ENABLED or signal.signal_type == SELL_SIGNAL:
        return signal
    previous_buy_signal = _resolve_previous_buy_signal(symbol, candles_df, signal.timeframe)
    if signal.signal_type == BUY_SIGNAL and previous_buy_signal is None:
        return _hold_signal_from(signal, "buy_waiting_confirmation")
    if previous_buy_signal is None:
        return signal
    try:
        latest_snapshot = build_greenwich_signal_snapshot(candles_df)
        latest_close = Decimal(str(candles_df["close"].iloc[-1]))
    except Exception:
        return signal
    if latest_close <= latest_snapshot.lower3:
        return SpotSignal(
            symbol,
            HOLD_SIGNAL,
            latest_close,
            latest_snapshot.close_time,
            "buy_confirmation_failed",
            signal.timeframe,
            str(latest_snapshot.close_time),
        )
    return SpotSignal(
        symbol,
        BUY_SIGNAL,
        latest_close,
        latest_snapshot.close_time,
        "greenwich_buy_confirmation",
        signal.timeframe,
        str(latest_snapshot.close_time),
    )


def _passes_anti_crash_buy_block(signal: SpotSignal, candles_df) -> bool:
    if not ANTI_CRASH_BUY_BLOCK_ENABLED or signal.signal_type != BUY_SIGNAL:
        return True
    if not _supports_candle_frame(candles_df) or "close" not in candles_df:
        return True
    try:
        required_length = ANTI_CRASH_LOOKBACK_CANDLES + 1
        if len(candles_df) < required_length:
            return True
        start_close = Decimal(str(candles_df["close"].iloc[-required_length]))
        current_close = Decimal(str(candles_df["close"].iloc[-1]))
    except Exception:
        return True
    if start_close <= 0:
        return True
    drop_ratio = (start_close - current_close) / start_close
    return drop_ratio <= ANTI_CRASH_MAX_DROP_RATIO


def _apply_anti_crash_buy_block(signal: SpotSignal, candles_df) -> SpotSignal:
    if _passes_anti_crash_buy_block(signal, candles_df):
        return signal
    return _hold_signal_from(signal, "buy_anti_crash_blocked")


def _apply_signal_filters(symbol: str, base_signal: SpotSignal, candles_df) -> SpotSignal:
    resolved_signal = _resolve_confirmation_buy_signal(symbol, base_signal, candles_df)
    resolved_signal = _apply_anti_crash_buy_block(resolved_signal, candles_df)
    return resolved_signal


class MultiTimeframeSpotPlanner:
    """Apply the D1 regime filter to H4 entry signals."""

    def __init__(self, d1_regime_filter_enabled: bool = True) -> None:
        self.d1_regime_filter_enabled = d1_regime_filter_enabled

    def plan(
        self,
        symbol: str,
        candles: dict,
        position_state: PositionState,
        available_quote_balance,
    ) -> SpotTradingPlan:
        """Build the H4 execution plan after applying the D1 regime filter."""

        d1_regime_blocked = self._is_d1_buy_blocked(candles["d1"])
        h4_signal = self._build_h4_signal(symbol, candles["h4"], position_state)
        resolved_signal = self._resolve(d1_regime_blocked, h4_signal)
        atr_size_multiplier = _resolve_atr_size_multiplier(candles["h4"])
        decision = decide_spot_execution(
            resolved_signal,
            position_state,
            available_quote_balance,
            atr_size_multiplier=atr_size_multiplier,
        )
        return SpotTradingPlan(signal=resolved_signal, decision=decision)

    def build_signal(self, symbol: str, candles: dict) -> MultiTimeframeSignal:
        """Return the raw H4 signal and resolved multi-timeframe signal."""

        d1_regime_blocked = self._is_d1_buy_blocked(candles["d1"])
        h4_signal = self._build_h4_signal(symbol, candles["h4"], PositionState(symbol, 0, 0, 0))
        resolved_signal = self._resolve(d1_regime_blocked, h4_signal)
        return MultiTimeframeSignal(
            symbol=symbol,
            d1_regime_blocked=d1_regime_blocked,
            h4=h4_signal,
            resolved=resolved_signal,
        )

    def _is_d1_buy_blocked(self, d1_candles_df) -> bool:
        if not self.d1_regime_filter_enabled or len(d1_candles_df) < 201:
            return False

        close = d1_candles_df["close"].astype("float64")
        fast = _wma(close, 50)
        slow = _wma(close, 200)
        death_cross = fast.shift(1).iloc[-1] >= slow.shift(1).iloc[-1] and fast.iloc[-1] < slow.iloc[-1]
        if not death_cross:
            return False
        adx = _calculate_adx(d1_candles_df)
        latest_adx = adx.iloc[-1]
        return bool(latest_adx >= 30)

    def _build_h4_signal(self, symbol: str, h4_candles_df, position_state: PositionState) -> SpotSignal:
        base_signal = generate_spot_signal(symbol, h4_candles_df, timeframe="h4")
        filtered_signal = _apply_signal_filters(symbol, base_signal, h4_candles_df)
        if filtered_signal.signal_type == SELL_SIGNAL:
            return filtered_signal
        if filtered_signal.signal_type != base_signal.signal_type or filtered_signal.reason != base_signal.reason:
            return filtered_signal
        if position_state.has_position and not position_state.first_take_profit_done:
            snapshot = build_greenwich_signal_snapshot(h4_candles_df)
            if snapshot.signal_high >= snapshot.upper1:
                return build_take_profit_signal(symbol, snapshot, timeframe="h4")
        return filtered_signal

    def _resolve(self, d1_regime_blocked: bool, h4: SpotSignal) -> SpotSignal:
        if d1_regime_blocked and h4.signal_type == BUY_SIGNAL:
            return SpotSignal(
                h4.symbol,
                HOLD_SIGNAL,
                h4.signal_price,
                h4.close_time,
                "d1_regime_blocks_h4_buy",
                h4.timeframe,
                h4.candle_id,
            )
        return h4


__all__ = ["GreenwichSpotPlanner", "MultiTimeframeSpotPlanner", "SpotTradingPlan"]
