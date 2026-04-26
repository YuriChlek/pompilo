from __future__ import annotations

import logging

from domain.cost_basis import minimum_exit_price
from domain.models import InventorySnapshot, LiveOrder, OrderSide, TargetOrder
from domain.strategy_config import StrategyConfig

logger = logging.getLogger(__name__)


def apply_execution_guardrails(
    symbol: str,
    current_orders: list[LiveOrder],
    target_orders: list[TargetOrder],
    inventory: InventorySnapshot,
    strategy_config: StrategyConfig,
) -> list[TargetOrder]:
    """Filter target orders through venue-safe execution guardrails for one symbol."""
    filtered_orders = [order for order in target_orders if order.symbol.upper() == symbol.upper() and order.size > 0]
    filtered_orders = filter_marketable_orders(filtered_orders, inventory, strategy_config)
    filtered_orders = dedupe_close_levels(filtered_orders, strategy_config.execution.min_level_distance_bps)
    filtered_orders = filter_no_loss_sells(filtered_orders, inventory, strategy_config)

    current_keys = {order_key(order.side, order.price, order.size) for order in current_orders}
    allowed_new_orders = strategy_config.execution.max_new_orders_per_cycle
    allowed_total_orders = strategy_config.execution.max_total_open_orders

    preserved_orders: list[TargetOrder] = []
    new_orders: list[TargetOrder] = []
    for order in filtered_orders:
        key = order_key(order.side, order.price, order.size)
        if key in current_keys:
            preserved_orders.append(order)
            continue
        if len(new_orders) >= allowed_new_orders:
            logger.info("execution_guardrail_skip_new_order symbol=%s reason=max_new_orders_per_cycle order=%s", symbol, order.client_order_id)
            continue
        new_orders.append(order)

    guarded_orders = preserved_orders + new_orders
    if len(guarded_orders) > allowed_total_orders:
        guarded_orders = guarded_orders[:allowed_total_orders]
        logger.info("execution_guardrail_trim_total_orders symbol=%s allowed=%s", symbol, allowed_total_orders)

    cancel_count = max(0, len(current_orders) - len(preserved_orders))
    if cancel_count > strategy_config.execution.max_cancel_replace_per_cycle:
        logger.info(
            "execution_guardrail_throttle_cancel_replace symbol=%s current=%s preserved=%s allowed=%s",
            symbol,
            len(current_orders),
            len(preserved_orders),
            strategy_config.execution.max_cancel_replace_per_cycle,
        )
        return preserved_orders
    logger.info(
        "execution_guardrail_result symbol=%s current=%s requested=%s guarded=%s new=%s",
        symbol,
        len(current_orders),
        len(target_orders),
        len(guarded_orders),
        len(new_orders),
    )
    return guarded_orders


def dedupe_close_levels(target_orders: list[TargetOrder], min_level_distance_bps: float) -> list[TargetOrder]:
    """Collapse same-side price levels that are closer than the configured minimum distance."""
    ordered = sorted(target_orders, key=lambda order: (order.side.value, order.price, order.client_order_id))
    deduped: list[TargetOrder] = []
    for order in ordered:
        previous = deduped[-1] if deduped else None
        if previous is None or previous.side != order.side:
            deduped.append(order)
            continue
        reference = max(order.price, previous.price, 1e-9)
        diff_bps = abs(order.price - previous.price) / reference * 10_000
        if diff_bps < min_level_distance_bps:
            keep = previous if previous.size >= order.size else order
            deduped[-1] = keep
            continue
        deduped.append(order)
    return deduped


def filter_no_loss_sells(
    target_orders: list[TargetOrder],
    inventory: InventorySnapshot,
    strategy_config: StrategyConfig,
) -> list[TargetOrder]:
    """Remove sell targets that violate the current no-loss exit floor."""
    if inventory.base_balance <= 0:
        return [order for order in target_orders if order.side != OrderSide.SELL]
    min_allowed_price = minimum_exit_price(inventory, strategy_config)
    if min_allowed_price is None:
        return [order for order in target_orders if order.side != OrderSide.SELL]
    return [
        order
        for order in target_orders
        if order.side != OrderSide.SELL or order.price >= min_allowed_price
    ]


def filter_marketable_orders(
    target_orders: list[TargetOrder],
    inventory: InventorySnapshot,
    strategy_config: StrategyConfig,
) -> list[TargetOrder]:
    """Reprice risky buy orders away from the market and drop unsafe sell orders."""
    live_price = inventory.mark_price
    if live_price <= 0:
        return target_orders
    min_buy_distance = strategy_config.execution.min_buy_distance_from_live_bps / 10_000
    min_sell_distance = strategy_config.execution.min_level_distance_bps / 10_000
    buy_orders = [order for order in target_orders if order.side == OrderSide.BUY]
    sell_orders = [order for order in target_orders if order.side == OrderSide.SELL]
    repriced_buy_orders = shift_buy_ladder_from_live_price(buy_orders, live_price, min_buy_distance)
    filtered: list[TargetOrder] = []
    filtered.extend(repriced_buy_orders)
    for order in sell_orders:
        if order.price <= live_price * (1 + min_sell_distance):
            logger.info(
                "execution_guardrail_skip_marketable_sell symbol=%s price=%s live_price=%s",
                order.symbol,
                order.price,
                live_price,
            )
            continue
        filtered.append(order)
    return filtered


def shift_buy_ladder_from_live_price(
    buy_orders: list[TargetOrder],
    live_price: float,
    min_buy_distance: float,
) -> list[TargetOrder]:
    """Shift the whole buy ladder down when the top buy level is too close to the market."""
    if not buy_orders:
        return []
    max_allowed_buy_price = live_price * (1 - min_buy_distance)
    highest_buy_price = max(order.price for order in buy_orders)
    if highest_buy_price < max_allowed_buy_price:
        return buy_orders

    shift_amount = highest_buy_price - max_allowed_buy_price
    logger.info(
        "execution_guardrail_shift_buy_ladder_from_live_offset symbol=%s levels=%s highest_buy_price=%s shifted_by=%s max_allowed_buy_price=%s live_price=%s",
        buy_orders[0].symbol,
        len(buy_orders),
        highest_buy_price,
        shift_amount,
        max_allowed_buy_price,
        live_price,
    )
    shifted_orders: list[TargetOrder] = []
    for order in buy_orders:
        adjusted_price = max(order.price - shift_amount, 1e-8)
        shifted_orders.append(
            TargetOrder(
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                price=adjusted_price,
                size=order.size,
                reduce_only=order.reduce_only,
                tag=order.tag,
            )
        )
    return shifted_orders


def order_key(side: OrderSide, price: float, size: float) -> tuple[str, float, float]:
    """Return a stable rounded key used for live-vs-target order comparisons."""
    return side.value, round(price, 8), round(size, 8)
