from __future__ import annotations

from domain.live_order_policy import live_to_target


def format_decision_dry_run(decision) -> str:
    """Render a compact structured dry-run diff against current live orders."""
    lines = [f"[{decision.symbol}] Regime: {decision.regime.value}"]
    current_orders = {
        (order.side.value, round(order.price, 8), round(order.size, 8)): order
        for order in live_to_target(decision.live_orders)
    }
    target_orders = {
        (order.side.value, round(order.price, 8), round(order.size, 8)): order
        for order in decision.target_orders
    }

    new_keys = sorted(target_orders.keys() - current_orders.keys())
    cancel_keys = sorted(current_orders.keys() - target_orders.keys())
    unchanged_keys = sorted(target_orders.keys() & current_orders.keys())

    for side, price, size in new_keys:
        lines.append(f"[{decision.symbol}] New {side} @ {price} x {size}")
    for side, price, size in cancel_keys:
        lines.append(f"[{decision.symbol}] Cancel {side} @ {price} x {size}")
    for side, price, size in unchanged_keys:
        lines.append(f"[{decision.symbol}] Keep {side} @ {price} x {size}")
    if len(lines) == 1:
        lines.append(f"[{decision.symbol}] No order changes")
    return "\n".join(lines)
