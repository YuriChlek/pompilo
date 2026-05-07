from __future__ import annotations

from dataclasses import dataclass

from domain.execution import decide_spot_execution
from domain.models import ExecutionDecision, MultiTimeframeSignal, PositionState, SpotSignal
from domain.signals import generate_spot_signal

BUY_SIGNAL = "buy"
HOLD_SIGNAL = "hold"


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

        signal = generate_spot_signal(symbol, candles_df)
        decision = decide_spot_execution(signal, position_state, available_quote_balance)
        return SpotTradingPlan(signal=signal, decision=decision)


def _wma(values, length: int):
    weights = list(range(1, length + 1))
    weight_sum = sum(weights)
    return values.rolling(length).apply(lambda window: sum(item * weight for item, weight in zip(window, weights)) / weight_sum, raw=True)


def _rma(values, length: int):
    return values.ewm(alpha=1 / length, adjust=False).mean()


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
        h4_signal = generate_spot_signal(symbol, candles["h4"], timeframe="h4")
        resolved_signal = self._resolve(d1_regime_blocked, h4_signal)
        decision = decide_spot_execution(resolved_signal, position_state, available_quote_balance)
        return SpotTradingPlan(signal=resolved_signal, decision=decision)

    def build_signal(self, symbol: str, candles: dict) -> MultiTimeframeSignal:
        """Return the raw H4 signal and resolved multi-timeframe signal."""

        d1_regime_blocked = self._is_d1_buy_blocked(candles["d1"])
        h4_signal = generate_spot_signal(symbol, candles["h4"], timeframe="h4")
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
