from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Optional

from utils.config import BUY_DIRECTION, SELL_DIRECTION

from .models import PendingOrder, Position, Trade


DECIMAL_ZERO = Decimal("0")
MARKET_PRIORITY_STRATEGY_MODES = {"trend_breakout"}


def _is_market_priority_strategy(strategy_mode: str) -> bool:
    """Return whether a strategy mode should override an existing pending limit order."""
    return str(strategy_mode) in MARKET_PRIORITY_STRATEGY_MODES


class ExecutionSimulator:
    """Simulate pending orders, open positions, and exits on replayed candles."""

    def __init__(self, allow_reversal: bool = True, intrabar_exit_priority: str = "stop"):
        """Initialize the local order and position execution simulator."""
        self.allow_reversal = allow_reversal
        self.intrabar_exit_priority = intrabar_exit_priority
        self.pending_order: Optional[PendingOrder] = None
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.filled_order_counts_by_strategy: Dict[str, int] = defaultdict(int)
        self.skipped_signal_counts: Dict[str, int] = defaultdict(int)
        self.skipped_signal_counts_by_strategy: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def queue_signal(self, signal: Optional[Dict[str, Any]], signal_bar_index: int) -> Optional[str]:
        """Queue a new signal for execution on the next candle."""
        if not signal:
            return None

        direction = str(signal["direction"]).lower()
        strategy_mode = str(signal.get("strategy_mode", "unknown"))
        if (
            self.pending_order
            and self.pending_order.order_type == "limit"
            and _is_market_priority_strategy(strategy_mode)
        ):
            self.pending_order = None

        if self.position and self.position.direction == direction:
            self._record_skipped_signal(strategy_mode, "same_direction_position")
            return "same_direction_position"
        if self.pending_order and str(self.pending_order.direction).lower() == direction:
            self._record_skipped_signal(strategy_mode, "same_direction_pending_order")
            return "same_direction_pending_order"
        if self.position and self.position.direction != direction and not self.allow_reversal:
            self._record_skipped_signal(strategy_mode, "reversal_disabled")
            return "reversal_disabled"

        reverse_existing_position = bool(
            self.position and self.position.direction != direction and self.allow_reversal
        )

        self.pending_order = PendingOrder(
            symbol=signal["symbol"],
            direction=direction,
            order_type=str(signal["order_type"]).lower(),
            requested_at=signal.get("time"),
            activate_on_bar=signal_bar_index + 1,
            price=Decimal(str(signal["price"])),
            stop_loss=Decimal(str(signal["stop_loss"])),
            take_profit=Decimal(str(signal["take_profit"])),
            signal_payload=dict(signal),
            reverse_existing_position=reverse_existing_position,
        )
        return "queued"

    def process_candle(self, candle: Dict[str, Any], bar_index: int) -> None:
        """Process one candle by attempting order fills first and exits second."""
        self._try_fill_pending_order(candle, bar_index)
        self._try_close_position(candle, bar_index)

    def finalize(self, candle: Dict[str, Any], bar_index: int) -> None:
        """Close any open position at the last available price at the end of history."""
        self.pending_order = None
        if not self.position:
            return
        self._close_position(
            exit_price=Decimal(str(candle["close"])),
            exit_time=candle.get("close_time") or candle.get("open_time"),
            exit_reason="end_of_data",
            closing_bar_index=bar_index,
        )

    def _try_fill_pending_order(self, candle: Dict[str, Any], bar_index: int) -> None:
        """Check whether the pending order can be filled on the current candle."""
        order = self.pending_order
        if not order or bar_index < order.activate_on_bar:
            return

        candle_open = Decimal(str(candle["open"]))
        candle_high = Decimal(str(candle["high"]))
        candle_low = Decimal(str(candle["low"]))
        fill_price: Optional[Decimal] = None

        if order.order_type == "market":
            fill_price = candle_open
        elif candle_low <= order.price <= candle_high:
            fill_price = order.price

        if fill_price is None:
            return

        if order.reverse_existing_position and self.position:
            self._close_position(
                exit_price=candle_open,
                exit_time=candle.get("open_time"),
                exit_reason="signal_reversal",
                closing_bar_index=bar_index,
            )

        self.position = Position(
            symbol=order.symbol,
            direction=order.direction,
            entry_time=candle.get("open_time"),
            entry_price=fill_price,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            opened_on_bar=bar_index,
            source_order_type=order.order_type,
            signal_payload=order.signal_payload,
        )
        strategy_mode = str(order.signal_payload.get("strategy_mode", "unknown"))
        self.filled_order_counts_by_strategy[strategy_mode] += 1
        self.pending_order = None

    def _try_close_position(self, candle: Dict[str, Any], bar_index: int) -> None:
        """Check whether the active position hit take-profit or stop-loss on this candle."""
        if not self.position:
            return

        candle_high = Decimal(str(candle["high"]))
        candle_low = Decimal(str(candle["low"]))
        position = self.position

        if position.direction == BUY_DIRECTION:
            hit_stop = candle_low <= position.stop_loss
            hit_target = candle_high >= position.take_profit
            stop_price = position.stop_loss
            target_price = position.take_profit
        else:
            hit_stop = candle_high >= position.stop_loss
            hit_target = candle_low <= position.take_profit
            stop_price = position.stop_loss
            target_price = position.take_profit

        if not hit_stop and not hit_target:
            return

        if hit_stop and hit_target:
            if self.intrabar_exit_priority == "target":
                exit_price = target_price
                exit_reason = "take_profit"
            else:
                exit_price = stop_price
                exit_reason = "stop_loss"
        elif hit_target:
            exit_price = target_price
            exit_reason = "take_profit"
        else:
            exit_price = stop_price
            exit_reason = "stop_loss"

        self._close_position(
            exit_price=exit_price,
            exit_time=candle.get("close_time") or candle.get("open_time"),
            exit_reason=exit_reason,
            closing_bar_index=bar_index,
        )

    def _close_position(
        self,
        *,
        exit_price: Decimal,
        exit_time: Any,
        exit_reason: str,
        closing_bar_index: int,
    ) -> None:
        """Record a closed position and append the finished trade to the journal."""
        if not self.position:
            return

        position = self.position
        pnl_pct = _calculate_pnl_pct(position.direction, position.entry_price, exit_price)
        initial_risk_distance = (
            Decimal(str(position.signal_payload.get("risk_distance")))
            if position.signal_payload.get("risk_distance") is not None
            else None
        )
        self.trades.append(
            Trade(
                symbol=position.symbol,
                strategy_mode=str(position.signal_payload.get("strategy_mode", "unknown")),
                direction=position.direction,
                entry_time=position.entry_time,
                exit_time=exit_time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                stop_loss=position.stop_loss,
                take_profit=position.take_profit,
                exit_reason=exit_reason,
                pnl_pct=pnl_pct,
                bars_held=max(0, closing_bar_index - position.opened_on_bar + 1),
                source_order_type=position.source_order_type,
                regime=str(position.signal_payload.get("regime", "unknown")),
                setup_type=str(position.signal_payload.get("setup_type", "unknown")),
                cluster=str(position.signal_payload.get("cluster", "other")),
                initial_risk_distance=initial_risk_distance,
                r_multiple=_calculate_r_multiple(
                    position.direction,
                    position.entry_price,
                    exit_price,
                    initial_risk_distance,
                ),
            )
        )
        self.position = None

    def _record_skipped_signal(self, strategy_mode: str, reason: str) -> None:
        """Track why a generated signal was not admitted to the execution queue."""
        self.skipped_signal_counts[reason] += 1
        self.skipped_signal_counts_by_strategy[strategy_mode][reason] += 1


def _calculate_pnl_pct(direction: str, entry_price: Decimal, exit_price: Decimal) -> Decimal:
    """Calculate percentage PnL for a long or short trade."""
    if entry_price == DECIMAL_ZERO:
        return DECIMAL_ZERO

    if direction == BUY_DIRECTION:
        raw_return = (exit_price - entry_price) / entry_price
    else:
        raw_return = (entry_price - exit_price) / entry_price

    return (raw_return * Decimal("100")).quantize(Decimal("0.01"))


def _calculate_r_multiple(
    direction: str,
    entry_price: Decimal,
    exit_price: Decimal,
    initial_risk_distance: Optional[Decimal],
) -> Optional[Decimal]:
    """Calculate realized R multiple from entry, exit, and initial risk distance."""
    if initial_risk_distance is None or initial_risk_distance <= DECIMAL_ZERO:
        return None

    if direction == BUY_DIRECTION:
        raw_r = (exit_price - entry_price) / initial_risk_distance
    else:
        raw_r = (entry_price - exit_price) / initial_risk_distance

    return raw_r.quantize(Decimal("0.01"))
