from __future__ import annotations

from decimal import Decimal, ROUND_DOWN

from bot_platform_service.trading_bots.spot_greenwich.domain.models import (
    GreenwichActionType,
    GreenwichExecutionConfig,
    GreenwichExecutionDecision,
    GreenwichPositionState,
    GreenwichSignalType,
    GreenwichSpotSignal,
)

DECIMAL_ZERO = Decimal("0")
PERCENT_BASE = Decimal("100")
PARTIAL_TAKE_PROFIT_REASON = "greenwich_take_profit_upper1"


def apply_portfolio_position_limit(
    decisions: dict[str, GreenwichExecutionDecision],
    position_states: dict[str, GreenwichPositionState],
    *,
    config: GreenwichExecutionConfig = GreenwichExecutionConfig(),
) -> dict[str, GreenwichExecutionDecision]:
    """Block new BUY entries when the portfolio already holds too many symbols."""

    if not config.portfolio_cap_enabled or config.portfolio_position_limit <= 0:
        return decisions

    active_positions_count = sum(1 for state in position_states.values() if state.has_position)
    remaining_slots = max(0, config.portfolio_position_limit - active_positions_count)
    opening_buy_symbols = [
        symbol
        for symbol, decision in decisions.items()
        if decision.action is GreenwichActionType.BUY and not position_states[symbol].has_position
    ]
    if not opening_buy_symbols:
        return decisions

    prioritized_symbols = sorted(opening_buy_symbols, key=lambda symbol: _portfolio_priority_key(symbol, config.portfolio_priority_symbols))
    allowed_symbols = set(prioritized_symbols[:remaining_slots])
    constrained_decisions: dict[str, GreenwichExecutionDecision] = {}
    for symbol, decision in decisions.items():
        if symbol not in opening_buy_symbols or symbol in allowed_symbols:
            constrained_decisions[symbol] = decision
            continue
        reason = "portfolio_position_limit_reached"
        if remaining_slots > 0:
            reason = "portfolio_position_limit_priority_blocked"
        constrained_decisions[symbol] = GreenwichExecutionDecision(
            GreenwichActionType.SKIP,
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
    signal: GreenwichSpotSignal,
    position_state: GreenwichPositionState,
    available_quote_balance: Decimal,
    *,
    config: GreenwichExecutionConfig = GreenwichExecutionConfig(),
    atr_size_multiplier: Decimal = Decimal("1"),
) -> GreenwichExecutionDecision:
    """Convert the current signal and position state into one pure decision."""

    if signal.signal_type is GreenwichSignalType.HOLD:
        return _skip(signal, "no_signal")

    if signal.signal_type is GreenwichSignalType.BUY:
        if position_state.has_position and signal.signal_price >= position_state.avg_entry_price:
            return _skip(signal, "buy_price_not_better_than_avg_entry")
        entry_count = _resolve_entry_count(position_state)
        if entry_count >= config.averaging_entry_limit:
            return _skip(signal, "max_entry_count_reached")
        quote_amount = _resolve_quote_amount_for_buy(
            position_state,
            available_quote_balance,
            config=config,
        ) * atr_size_multiplier
        if quote_amount <= 0:
            return _skip(signal, "insufficient_quote_balance")
        quantity = _quantize_quantity(quote_amount / signal.signal_price)
        if quantity <= 0:
            return _skip(signal, "buy_quantity_too_small")
        return GreenwichExecutionDecision(
            GreenwichActionType.BUY,
            signal.symbol,
            signal.signal_price,
            quantity,
            quote_amount,
            "greenwich_accumulation_buy",
            signal.timeframe,
            signal.candle_id,
        )

    if not position_state.has_position:
        return _skip(signal, "no_position_to_sell")

    min_sell_price = position_state.avg_entry_price * (Decimal("1") + config.min_profit_ratio)
    if signal.signal_price < min_sell_price:
        return _skip(signal, "sell_price_not_profitable")
    if signal.reason == PARTIAL_TAKE_PROFIT_REASON:
        quantity = _quantize_quantity(position_state.quantity / Decimal("2"))
        if quantity <= 0:
            return _skip(signal, "partial_sell_quantity_too_small")
        quote_amount = quantity * signal.signal_price
        return GreenwichExecutionDecision(
            GreenwichActionType.SELL,
            signal.symbol,
            signal.signal_price,
            quantity,
            quote_amount,
            PARTIAL_TAKE_PROFIT_REASON,
            signal.timeframe,
            signal.candle_id,
        )
    quote_amount = position_state.quantity * signal.signal_price
    return GreenwichExecutionDecision(
        GreenwichActionType.SELL,
        signal.symbol,
        signal.signal_price,
        position_state.quantity,
        quote_amount,
        "greenwich_profitable_exit",
        signal.timeframe,
        signal.candle_id,
    )


def _skip(signal: GreenwichSpotSignal, reason: str) -> GreenwichExecutionDecision:
    return GreenwichExecutionDecision(
        GreenwichActionType.SKIP,
        signal.symbol,
        signal.signal_price,
        DECIMAL_ZERO,
        DECIMAL_ZERO,
        reason,
        signal.timeframe,
        signal.candle_id,
    )


def _quantize_quantity(quantity: Decimal) -> Decimal:
    return quantity.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)


def _quote_amount_from_balance(
    available_quote_balance: Decimal,
    *,
    config: GreenwichExecutionConfig,
) -> Decimal:
    return (available_quote_balance * config.deposit_percent) / PERCENT_BASE


def _resolve_entry_count(position_state: GreenwichPositionState) -> int:
    if not position_state.has_position:
        return 0
    return max(0, int(position_state.entry_count))


def _resolve_buy_size_percent(
    entry_count: int,
    *,
    config: GreenwichExecutionConfig,
) -> Decimal:
    if entry_count <= 0:
        return PERCENT_BASE
    if entry_count == 1:
        return config.averaging_entry_2_size_percent
    if entry_count == 2:
        return config.averaging_entry_3_size_percent
    return DECIMAL_ZERO


def _resolve_quote_amount_for_buy(
    position_state: GreenwichPositionState,
    available_quote_balance: Decimal,
    *,
    config: GreenwichExecutionConfig,
) -> Decimal:
    entry_count = _resolve_entry_count(position_state)
    if entry_count >= config.averaging_entry_limit:
        return DECIMAL_ZERO
    base_quote_amount = _quote_amount_from_balance(available_quote_balance, config=config)
    size_percent = _resolve_buy_size_percent(entry_count, config=config)
    return (base_quote_amount * size_percent) / PERCENT_BASE


def _portfolio_priority_key(symbol: str, priority_symbols: tuple[str, ...]) -> tuple[int, str]:
    normalized_symbol = symbol.upper()
    try:
        return (priority_symbols.index(normalized_symbol), normalized_symbol)
    except ValueError:
        return (len(priority_symbols), normalized_symbol)

