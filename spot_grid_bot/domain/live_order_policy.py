from __future__ import annotations

import re

from domain.models import LiveOrder, OrderSide, TargetOrder

BOT_ORDER_LINK_ID_RE = re.compile(r"^[a-z0-9-]{1,16}-[0-9a-f]{16,32}$")


def is_entry_order_tag(tag: str) -> bool:
    """Return whether an order tag represents a buy-side entry ladder order."""
    normalized = tag.strip().lower()
    return normalized in {"range_buy", "trend_pullback_buy", "live_open_order_buy"}


def is_bot_managed_order_link_id(client_order_id: str) -> bool:
    """Return whether a live order link id appears to be created by this bot."""
    normalized = (client_order_id or "").strip().lower()
    return bool(normalized) and BOT_ORDER_LINK_ID_RE.match(normalized) is not None


def is_bot_managed_live_order(order: LiveOrder) -> bool:
    """Return whether a live order appears to be bot-managed."""
    return is_bot_managed_order_link_id(order.client_order_id)


def has_live_entry_orders(live_orders: list[LiveOrder]) -> bool:
    """Return whether the live order set still contains buy-side entry orders."""
    return any(
        order.side == OrderSide.BUY
        and order.size - order.filled_size > 0
        and is_bot_managed_live_order(order)
        for order in live_orders
    )


def live_to_target(live_orders: list[LiveOrder]) -> list[TargetOrder]:
    """Convert live exchange orders into target-order shape for no-op decisions."""
    return [
        TargetOrder(
            client_order_id=order.client_order_id or order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=order.price,
            size=order.size,
            reduce_only=order.side == OrderSide.SELL,
            tag="live_open_order_sell" if order.side == OrderSide.SELL else "live_open_order_buy",
        )
        for order in live_orders
    ]
