from __future__ import annotations

from decimal import Decimal, ROUND_DOWN

from domain.models import ExecutionDecision, PositionState, SpotSignal
from utils.config import BUY_SIGNAL, HOLD_SIGNAL, MIN_PROFIT_RATIO, ORDER_DEPOSIT_PERCENT, SELL_SIGNAL

DECIMAL_ZERO = Decimal("0")
PERCENT_BASE = Decimal("100")


def _quantize_quantity(quantity: Decimal) -> Decimal:
    return quantity.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


def _quote_amount_from_balance(available_quote_balance: Decimal) -> Decimal:
    return (available_quote_balance * ORDER_DEPOSIT_PERCENT) / PERCENT_BASE


def decide_spot_execution(
    signal: SpotSignal,
    position_state: PositionState,
    available_quote_balance: Decimal,
) -> ExecutionDecision:
    """Convert the current signal and position state into one execution decision."""

    if signal.signal_type == HOLD_SIGNAL:
        return ExecutionDecision("skip", signal.symbol, signal.signal_price, DECIMAL_ZERO, DECIMAL_ZERO, "no_signal")

    if signal.signal_type == BUY_SIGNAL:
        if position_state.has_position and signal.signal_price >= position_state.avg_entry_price:
            return ExecutionDecision(
                "skip",
                signal.symbol,
                signal.signal_price,
                DECIMAL_ZERO,
                DECIMAL_ZERO,
                "buy_price_not_better_than_avg_entry",
            )
        quote_amount = _quote_amount_from_balance(available_quote_balance)
        if quote_amount <= 0:
            return ExecutionDecision(
                "skip",
                signal.symbol,
                signal.signal_price,
                DECIMAL_ZERO,
                DECIMAL_ZERO,
                "insufficient_quote_balance",
            )
        quantity = _quantize_quantity(quote_amount / signal.signal_price)
        if quantity <= 0:
            return ExecutionDecision(
                "skip",
                signal.symbol,
                signal.signal_price,
                DECIMAL_ZERO,
                DECIMAL_ZERO,
                "buy_quantity_too_small",
            )
        return ExecutionDecision(
            "buy",
            signal.symbol,
            signal.signal_price,
            quantity,
            quote_amount,
            "greenwich_accumulation_buy",
        )

    if not position_state.has_position:
        return ExecutionDecision("skip", signal.symbol, signal.signal_price, DECIMAL_ZERO, DECIMAL_ZERO, "no_position_to_sell")

    min_sell_price = position_state.avg_entry_price * (Decimal("1") + MIN_PROFIT_RATIO)
    if signal.signal_price <= min_sell_price:
        return ExecutionDecision(
            "skip",
            signal.symbol,
            signal.signal_price,
            DECIMAL_ZERO,
            DECIMAL_ZERO,
            "sell_price_not_profitable",
        )
    quote_amount = position_state.quantity * signal.signal_price
    return ExecutionDecision(
        "sell",
        signal.symbol,
        signal.signal_price,
        position_state.quantity,
        quote_amount,
        "greenwich_profitable_exit",
    )


__all__ = ["decide_spot_execution"]
