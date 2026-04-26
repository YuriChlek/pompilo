from __future__ import annotations

from domain.indicators import IndicatorSnapshot
from domain.models import InventorySnapshot
from domain.strategy_config import StrategyConfig


def effective_reference_price(inventory: InventorySnapshot) -> float | None:
    """Return the best available inventory reference price for exit-planning logic."""
    if inventory.base_balance > 0:
        reference_price = inventory.cost_basis_price
    else:
        reference_price = inventory.mark_price
    if reference_price is None:
        return None
    if reference_price <= 0:
        return None
    return reference_price


def minimum_exit_price(
    inventory: InventorySnapshot,
    strategy_config: StrategyConfig,
) -> float | None:
    """Return the minimum allowed no-loss exit price for the current inventory snapshot."""
    reference_price = effective_reference_price(inventory)
    if reference_price is None:
        return None
    return reference_price * (1 + strategy_config.execution.min_sell_markup_bps / 10_000)


def minimum_take_profit_price(
    inventory: InventorySnapshot,
    indicators: IndicatorSnapshot,
    strategy_config: StrategyConfig,
) -> float | None:
    """Return the minimum take-profit floor based on cost basis and ATR expansion."""
    markup_floor = minimum_exit_price(inventory, strategy_config)
    if markup_floor is None:
        return None

    if inventory.base_balance <= 0:
        return markup_floor

    reference_price = effective_reference_price(inventory)
    if reference_price is None:
        return markup_floor
    atr_profit_floor = reference_price + indicators.atr14 * strategy_config.grid.rebalance_profit_target_atr
    return max(markup_floor, atr_profit_floor)


def rebased_take_profit_price(
    target_price: float,
    level_index: int,
    inventory: InventorySnapshot,
    indicators: IndicatorSnapshot,
    strategy_config: StrategyConfig,
    *,
    tick_size: float | None = None,
) -> float:
    """Raise a sell target to the minimum profitable ladder level when needed."""
    take_profit_floor = minimum_take_profit_price(inventory, indicators, strategy_config)
    if take_profit_floor is None:
        return target_price

    step = max(
        indicators.atr14 * strategy_config.grid.atr_grid_step_multiplier,
        tick_size or 0.0,
        1e-8,
    )
    return max(target_price, take_profit_floor + step * max(level_index, 0))
