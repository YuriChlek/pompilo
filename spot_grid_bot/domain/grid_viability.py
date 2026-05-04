from __future__ import annotations

from dataclasses import replace

from domain.models import TargetOrder, VenueConstraints


def normalize_price_to_tick(price: float, tick_size: float | None) -> float:
    """Round a price to the nearest symbol tick when a tick size is available."""
    if tick_size is None or tick_size <= 0:
        return round(price, 8)
    return round(round(price / tick_size) * tick_size, 8)


def apply_venue_viability(
    target_orders: list[TargetOrder],
    venue_constraints: VenueConstraints | None,
) -> list[TargetOrder]:
    """Normalize prices and collapse duplicate symbol/side levels before execution."""
    if venue_constraints is None:
        return target_orders

    tick_size = venue_constraints.tick_size
    merged: dict[tuple[str, str, float, bool], TargetOrder] = {}
    for order in target_orders:
        normalized_price = normalize_price_to_tick(order.price, tick_size)
        key = (order.symbol.upper(), order.side.value, normalized_price, order.reduce_only)
        existing = merged.get(key)
        if existing is None:
            merged[key] = replace(order, price=normalized_price)
            continue
        merged[key] = replace(
            existing,
            size=round(existing.size + order.size, 8),
            tag=_merge_order_tags(existing.tag, order.tag),
        )

    return sorted(
        merged.values(),
        key=lambda order: (order.side.value, order.price, order.client_order_id),
    )


def _merge_order_tags(existing_tag: str, new_tag: str) -> str:
    """Preserve both source tags when normalized levels collapse into one order."""
    if not existing_tag:
        return new_tag
    if not new_tag or new_tag == existing_tag:
        return existing_tag
    return f"{existing_tag}+{new_tag}"
