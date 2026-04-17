from __future__ import annotations

from domain.models import LiveOrder, TargetOrder


def target_orders_diff_count(
    live_orders: list[LiveOrder],
    target_orders: list[TargetOrder],
    *,
    price_diff_bps: float,
    size_diff_ratio: float,
) -> int:
    unmatched_live = list(live_orders)
    unmatched_target: list[TargetOrder] = []

    for target in target_orders:
        match_index = _find_matching_live_order(
            unmatched_live,
            target,
            price_diff_bps=price_diff_bps,
            size_diff_ratio=size_diff_ratio,
        )
        if match_index is None:
            unmatched_target.append(target)
            continue
        unmatched_live.pop(match_index)

    return len(unmatched_live) + len(unmatched_target)


def _find_matching_live_order(
    live_orders: list[LiveOrder],
    target: TargetOrder,
    *,
    price_diff_bps: float,
    size_diff_ratio: float,
) -> int | None:
    for index, live in enumerate(live_orders):
        if live.symbol.upper() != target.symbol.upper():
            continue
        if live.side != target.side:
            continue
        if not _price_matches(live.price, target.price, price_diff_bps):
            continue
        if not _size_matches(live.size, target.size, size_diff_ratio):
            continue
        return index
    return None


def _price_matches(live_price: float, target_price: float, price_diff_bps: float) -> bool:
    reference = max(abs(live_price), abs(target_price), 1e-9)
    diff_bps = abs(live_price - target_price) / reference * 10_000
    return diff_bps <= price_diff_bps


def _size_matches(live_size: float, target_size: float, size_diff_ratio: float) -> bool:
    reference = max(abs(live_size), abs(target_size), 1e-9)
    return abs(live_size - target_size) / reference <= size_diff_ratio
