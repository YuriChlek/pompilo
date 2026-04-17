from __future__ import annotations

from domain.cost_basis import effective_reference_price, minimum_exit_price
from domain.models import DeRiskMode, InventorySnapshot, OrderSide, TargetOrder
from domain.strategy_config import StrategyConfig


def build_de_risk_orders(
    *,
    symbol: str,
    inventory: InventorySnapshot,
    live_price: float,
    strategy_config: StrategyConfig,
    mode: DeRiskMode,
) -> list[TargetOrder]:
    """Build reduce-only sell orders for staged de-risking without violating no-loss rules."""
    if inventory.base_balance <= 0 or mode == DeRiskMode.NONE:
        return []

    reference_price = effective_reference_price(inventory)
    min_allowed_price = minimum_exit_price(inventory, strategy_config)
    if reference_price is None or min_allowed_price is None:
        return []

    if live_price < min_allowed_price:
        return []

    size_ratio = {
        DeRiskMode.SOFT: 0.20,
        DeRiskMode.HARD: 0.35,
        DeRiskMode.PANIC: 0.50,
    }.get(mode, 0.0)
    target_size = max(
        round(inventory.base_balance * size_ratio / strategy_config.execution.size_step) * strategy_config.execution.size_step,
        strategy_config.execution.min_order_size,
    )
    target_size = min(target_size, inventory.base_balance)
    if target_size <= 0:
        return []

    return [
        TargetOrder(
            client_order_id=f"derisk-{mode.value.lower()}",
            symbol=symbol,
            side=OrderSide.SELL,
            price=round(max(live_price, min_allowed_price), 8),
            size=target_size,
            reduce_only=True,
            tag=f"derisk_{mode.value.lower()}",
        )
    ]
