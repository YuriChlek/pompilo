from __future__ import annotations

from dataclasses import dataclass

from domain.models import LiveOrder, OrderSide


@dataclass(slots=True, frozen=True)
class OutstandingExposure:
    outstanding_buy_notional: float
    outstanding_sell_size: float


def calculate_outstanding_exposure(live_orders: list[LiveOrder]) -> OutstandingExposure:
    buy_notional = 0.0
    sell_size = 0.0
    for order in live_orders:
        remaining_size = max(order.size - order.filled_size, 0.0)
        if remaining_size <= 0:
            continue
        if order.side == OrderSide.BUY:
            buy_notional += remaining_size * order.price
        else:
            sell_size += remaining_size
    return OutstandingExposure(
        outstanding_buy_notional=buy_notional,
        outstanding_sell_size=sell_size,
    )
