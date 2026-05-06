from __future__ import annotations

from decimal import Decimal
from typing import Any

from trading.domain.models import SignalDirection, to_domain_signal_direction


def should_fill_dry_run(
    signal_direction,
    entry_price: Decimal | None,
    best_bid: Decimal | None,
    best_ask: Decimal | None,
) -> bool:
    signal_direction = to_domain_signal_direction(signal_direction)
    if entry_price is None:
        return False
    if signal_direction == SignalDirection.LONG:
        return best_ask is not None and best_ask <= entry_price
    if signal_direction == SignalDirection.SHORT:
        return best_bid is not None and best_bid >= entry_price
    return False


def pending_entry_invalidation_reason(
    current_signal_reason: str,
    current_signal_direction,
    pending_signal_direction,
    has_pending_wall: bool,
    wall_is_active: bool | None,
    has_reference_book: bool,
) -> str | None:
    current_signal_direction = to_domain_signal_direction(current_signal_direction)
    pending_signal_direction = to_domain_signal_direction(pending_signal_direction)
    if current_signal_reason in {"missing_analysis_book", "stale_analysis_book"}:
        return None

    if current_signal_direction not in {SignalDirection.NONE, pending_signal_direction}:
        return "signal_reversed"

    if not has_pending_wall:
        return None

    if not has_reference_book:
        return None

    if wall_is_active is False:
        return "wall_disappeared"

    return None


def current_market_price(
    position_side: str,
    best_bid: Decimal | None,
    best_ask: Decimal | None,
    fallback_price: Decimal,
) -> Decimal:
    if best_bid is None or best_ask is None:
        return fallback_price
    return best_bid if position_side.lower() == "buy" else best_ask


def best_price_seen(position_side: str, previous_best_price_seen: Decimal, market_price: Decimal) -> Decimal:
    if position_side.lower() == "buy":
        return max(previous_best_price_seen, market_price)
    return min(previous_best_price_seen, market_price)


def break_even_stop_price(
    side: str,
    entry_price: Decimal,
    current_price: Decimal,
    tick_size: Decimal,
    arm_ticks: int,
    buffer_ticks: int,
) -> Decimal | None:
    if tick_size <= 0:
        return None

    arm_distance = tick_size * Decimal(arm_ticks)
    buffer_distance = tick_size * Decimal(buffer_ticks)
    required_distance = max(arm_distance, buffer_distance + tick_size)
    side_lower = side.lower()

    if side_lower == "buy":
        if current_price < entry_price + required_distance:
            return None
        return _quantize_price(entry_price + buffer_distance)

    if current_price > entry_price - required_distance:
        return None
    return _quantize_price(entry_price - buffer_distance)


def stop_improves(side: str, current_stop: Decimal, new_stop: Decimal) -> bool:
    if side.lower() == "buy":
        return new_stop > current_stop
    return new_stop < current_stop


def position_exit_reason(
    position_side: str,
    stop_price: Decimal,
    take_profit_price: Decimal,
    current_signal_direction,
    market_price: Decimal,
) -> str | None:
    current_signal_direction = to_domain_signal_direction(current_signal_direction)
    if position_side.lower() == "buy":
        if market_price <= stop_price:
            return "stop_loss"
        if market_price >= take_profit_price:
            return "take_profit"
        position_direction = SignalDirection.LONG
    else:
        if market_price >= stop_price:
            return "stop_loss"
        if market_price <= take_profit_price:
            return "take_profit"
        position_direction = SignalDirection.SHORT

    if current_signal_direction not in {SignalDirection.NONE, position_direction}:
        return "signal_reversal"

    return None


def infer_exchange_exit_reason(
    position_side: str,
    stop_price: Decimal,
    take_profit_price: Decimal,
    market_price: Decimal,
) -> str:
    if position_side.lower() == "buy":
        if market_price >= take_profit_price:
            return "take_profit"
        if market_price <= stop_price:
            return "stop_loss"
    else:
        if market_price <= take_profit_price:
            return "take_profit"
        if market_price >= stop_price:
            return "stop_loss"
    return "position_closed_on_exchange"


def to_decimal_price(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(value))


def _quantize_price(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.00000001"))
