from __future__ import annotations

from decimal import Decimal, ROUND_DOWN

from domain.models import ExecutionDecision, PositionState, SpotSignal
from utils.config import (
    AVERAGING_ENTRY_2_SIZE_PERCENT,
    AVERAGING_ENTRY_3_SIZE_PERCENT,
    AVERAGING_ENTRY_LIMIT,
    BUY_SIGNAL,
    HOLD_SIGNAL,
    MIN_PROFIT_RATIO,
    ORDER_DEPOSIT_PERCENT,
    SELL_SIGNAL,
)

DECIMAL_ZERO = Decimal("0")
PERCENT_BASE = Decimal("100")
PARTIAL_TAKE_PROFIT_REASON = "greenwich_take_profit_upper1"


def _quantize_quantity(quantity: Decimal) -> Decimal:
    return quantity.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


def _quote_amount_from_balance(available_quote_balance: Decimal) -> Decimal:
    return (available_quote_balance * ORDER_DEPOSIT_PERCENT) / PERCENT_BASE


def _resolve_entry_count(position_state: PositionState) -> int:
    if not position_state.has_position:
        return 0
    return max(0, int(position_state.entry_count))


def _resolve_buy_size_percent(entry_count: int) -> Decimal:
    if entry_count <= 0:
        return PERCENT_BASE
    if entry_count == 1:
        return AVERAGING_ENTRY_2_SIZE_PERCENT
    if entry_count == 2:
        return AVERAGING_ENTRY_3_SIZE_PERCENT
    return DECIMAL_ZERO


def _resolve_quote_amount_for_buy(position_state: PositionState, available_quote_balance: Decimal) -> Decimal:
    entry_count = _resolve_entry_count(position_state)
    if entry_count >= AVERAGING_ENTRY_LIMIT:
        return DECIMAL_ZERO
    base_quote_amount = _quote_amount_from_balance(available_quote_balance)
    size_percent = _resolve_buy_size_percent(entry_count)
    return (base_quote_amount * size_percent) / PERCENT_BASE


def _portfolio_priority_key(symbol: str, priority_symbols: tuple[str, ...]) -> tuple[int, str]:
    normalized_symbol = symbol.upper()
    try:
        return (priority_symbols.index(normalized_symbol), normalized_symbol)
    except ValueError:
        return (len(priority_symbols), normalized_symbol)


def apply_portfolio_position_limit(
    decisions: dict[str, ExecutionDecision],
    position_states: dict[str, PositionState],
    *,
    position_limit: int,
    priority_symbols: tuple[str, ...] = (),
) -> dict[str, ExecutionDecision]:
    """Block new BUY entries when the portfolio already holds too many symbols."""

    if position_limit <= 0:
        return decisions

    active_positions_count = sum(1 for state in position_states.values() if state.has_position)
    remaining_slots = max(0, position_limit - active_positions_count)
    opening_buy_symbols = [
        symbol
        for symbol, decision in decisions.items()
        if decision.action == BUY_SIGNAL and not position_states[symbol].has_position
    ]
    if not opening_buy_symbols:
        return decisions

    prioritized_symbols = sorted(opening_buy_symbols, key=lambda symbol: _portfolio_priority_key(symbol, priority_symbols))
    allowed_symbols = set(prioritized_symbols[:remaining_slots])
    constrained_decisions: dict[str, ExecutionDecision] = {}
    for symbol, decision in decisions.items():
        if symbol not in opening_buy_symbols or symbol in allowed_symbols:
            constrained_decisions[symbol] = decision
            continue
        reason = "portfolio_position_limit_reached"
        if remaining_slots > 0:
            reason = "portfolio_position_limit_priority_blocked"
        constrained_decisions[symbol] = ExecutionDecision(
            "skip",
            decision.symbol,
            decision.signal_price,
            DECIMAL_ZERO,
            DECIMAL_ZERO,
            reason,
            decision.signal_timeframe,
            decision.signal_candle_id,
        )
    return constrained_decisions


def decide_spot_execution(
    signal: SpotSignal,
    position_state: PositionState,
    available_quote_balance: Decimal,
    *,
    atr_size_multiplier: Decimal = Decimal("1"),
) -> ExecutionDecision:
    """Convert the current signal and position state into one execution decision."""

    if signal.signal_type == HOLD_SIGNAL:
        return ExecutionDecision("skip", signal.symbol, signal.signal_price, DECIMAL_ZERO, DECIMAL_ZERO, "no_signal", signal.timeframe, signal.candle_id)

    if signal.signal_type == BUY_SIGNAL:
        if position_state.has_position and signal.signal_price >= position_state.avg_entry_price:
            return ExecutionDecision(
                "skip",
                signal.symbol,
                signal.signal_price,
                DECIMAL_ZERO,
                DECIMAL_ZERO,
                "buy_price_not_better_than_avg_entry",
                signal.timeframe,
                signal.candle_id,
            )
        entry_count = _resolve_entry_count(position_state)
        if entry_count >= AVERAGING_ENTRY_LIMIT:
            return ExecutionDecision(
                "skip",
                signal.symbol,
                signal.signal_price,
                DECIMAL_ZERO,
                DECIMAL_ZERO,
                "max_entry_count_reached",
                signal.timeframe,
                signal.candle_id,
            )
        quote_amount = _resolve_quote_amount_for_buy(position_state, available_quote_balance) * atr_size_multiplier
        if quote_amount <= 0:
            return ExecutionDecision(
                "skip",
                signal.symbol,
                signal.signal_price,
                DECIMAL_ZERO,
                DECIMAL_ZERO,
                "insufficient_quote_balance",
                signal.timeframe,
                signal.candle_id,
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
                signal.timeframe,
                signal.candle_id,
            )
        return ExecutionDecision(
            "buy",
            signal.symbol,
            signal.signal_price,
            quantity,
            quote_amount,
            "greenwich_accumulation_buy",
            signal.timeframe,
            signal.candle_id,
        )

    if not position_state.has_position:
        return ExecutionDecision("skip", signal.symbol, signal.signal_price, DECIMAL_ZERO, DECIMAL_ZERO, "no_position_to_sell", signal.timeframe, signal.candle_id)

    min_sell_price = position_state.avg_entry_price * (Decimal("1") + MIN_PROFIT_RATIO)
    if signal.signal_price < min_sell_price:
        return ExecutionDecision(
            "skip",
            signal.symbol,
            signal.signal_price,
            DECIMAL_ZERO,
            DECIMAL_ZERO,
            "sell_price_not_profitable",
            signal.timeframe,
            signal.candle_id,
        )
    if signal.reason == PARTIAL_TAKE_PROFIT_REASON:
        quantity = _quantize_quantity(position_state.quantity / Decimal("2"))
        if quantity <= 0:
            return ExecutionDecision(
                "skip",
                signal.symbol,
                signal.signal_price,
                DECIMAL_ZERO,
                DECIMAL_ZERO,
                "partial_sell_quantity_too_small",
                signal.timeframe,
                signal.candle_id,
            )
        quote_amount = quantity * signal.signal_price
        return ExecutionDecision(
            "sell",
            signal.symbol,
            signal.signal_price,
            quantity,
            quote_amount,
            PARTIAL_TAKE_PROFIT_REASON,
            signal.timeframe,
            signal.candle_id,
        )
    quote_amount = position_state.quantity * signal.signal_price
    return ExecutionDecision(
        "sell",
        signal.symbol,
        signal.signal_price,
        position_state.quantity,
        quote_amount,
        "greenwich_profitable_exit",
        signal.timeframe,
        signal.candle_id,
    )


__all__ = ["apply_portfolio_position_limit", "decide_spot_execution"]
