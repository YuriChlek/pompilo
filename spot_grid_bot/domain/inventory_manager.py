from __future__ import annotations

import logging

from domain.allocation import reserve_quote_amount
from domain.models import GridSpec, InventorySnapshot, OrderSide, RegimeType, VenueConstraints
from domain.strategy_config import StrategyConfig

logger = logging.getLogger(__name__)


class InventoryManager:
    """Allocate grid levels against quote capital and current base inventory."""

    def __init__(self, config: StrategyConfig) -> None:
        """Store strategy configuration used for inventory sizing."""
        self.config = config

    def reserve_quote(self, inventory: InventorySnapshot) -> float:
        """Return the quote balance that should remain reserved and not be allocated."""
        return reserve_quote_amount(inventory, self.config)

    def can_accumulate(self, inventory: InventorySnapshot) -> bool:
        """Check whether current balances still allow new buy-side accumulation."""
        inventory_cap = inventory.total_equity * self.config.risk.max_symbol_inventory_pct_of_equity
        return (
            inventory.inventory_notional < inventory_cap
            and inventory.inventory_notional < self.config.risk.max_symbol_notional_cap
            and inventory.available_quote >= self.config.risk.min_quote_balance
        )

    def allocate_grid(
        self,
        grid: GridSpec,
        inventory: InventorySnapshot,
        symbol_entry_budget: float,
        venue_constraints: VenueConstraints | None = None,
    ) -> GridSpec:
        """Apply size and notional allocation to a grid using current inventory constraints."""
        inventory.reserved_quote = self.reserve_quote(inventory)
        total_allocatable = min(inventory.available_quote, max(symbol_entry_budget, 0.0))
        buy_levels = [level for level in grid.levels if level.side == OrderSide.BUY]
        sell_levels = [level for level in grid.levels if level.side == OrderSide.SELL]
        remaining_inventory_capacity = min(
            max(inventory.total_equity * self.config.risk.max_symbol_inventory_pct_of_equity - inventory.inventory_notional, 0.0),
            max(self.config.risk.max_symbol_notional_cap - inventory.inventory_notional, 0.0),
            max(self.config.risk.max_inventory_notional - inventory.inventory_notional, 0.0),
        )
        effective_budget = min(total_allocatable, remaining_inventory_capacity)
        viable_buy_levels = self._viable_buy_level_count(buy_levels, effective_budget, venue_constraints)
        buy_level_weights = self._buy_level_weights(grid.regime, viable_buy_levels)
        spent_budget = 0.0
        viable_sell_levels = self._viable_sell_level_count(sell_levels, inventory, venue_constraints)
        sell_level_weights = self._sell_level_weights(grid.regime, viable_sell_levels, inventory)
        if viable_buy_levels < len(buy_levels):
            logger.info(
                "planner_buy_levels_reduced_due_to_budget total_buy_levels=%s viable_buy_levels=%s effective_budget=%s min_order_notional=%s",
                len(buy_levels),
                viable_buy_levels,
                round(effective_budget, 2),
                self._effective_min_order_notional(venue_constraints),
            )
        if viable_sell_levels < len(sell_levels):
            logger.info(
                "planner_sell_levels_reduced_due_to_venue_constraints total_sell_levels=%s viable_sell_levels=%s base_balance=%s min_order_qty=%s min_order_amt=%s",
                len(sell_levels),
                viable_sell_levels,
                inventory.base_balance,
                venue_constraints.min_order_qty if venue_constraints is not None else 0.0,
                venue_constraints.min_order_amt if venue_constraints is not None else 0.0,
            )
        assigned_sell_levels = 0
        assigned_buy_levels = 0

        for level in grid.levels:
            if level.side == OrderSide.SELL:
                if assigned_sell_levels >= viable_sell_levels:
                    level.size = 0.0
                    level.notional = 0.0
                    continue
                sell_weight = sell_level_weights[assigned_sell_levels] if assigned_sell_levels < len(sell_level_weights) else 0.0
                level.size = max(min(inventory.base_balance * sell_weight, inventory.base_balance), 0.0)
                level.size = self._normalize_size(level.size, venue_constraints.qty_step if venue_constraints is not None else None)
                level.notional = self._round_notional(level.size * level.price)
                if level.size <= 0 or level.notional <= 0:
                    level.size = 0.0
                    level.notional = 0.0
                    continue
                assigned_sell_levels += 1
                continue

            if not self.can_accumulate(inventory) or spent_budget >= effective_budget:
                level.size = 0.0
                level.notional = 0.0
                continue
            if assigned_buy_levels >= viable_buy_levels:
                level.size = 0.0
                level.notional = 0.0
                continue

            weight = buy_level_weights[assigned_buy_levels] if assigned_buy_levels < len(buy_level_weights) else 0.0
            level_budget = min(
                self.config.risk.max_notional_per_level,
                effective_budget * weight,
                effective_budget - spent_budget,
            )
            level_budget = self._round_notional(level_budget)
            size = self._normalize_size(
                level_budget / level.price,
                venue_constraints.qty_step if venue_constraints is not None else None,
            )
            notional = self._round_notional(size * level.price)
            if (
                size < self.config.execution.min_order_size
                or notional <= 0
            ):
                level.size = 0.0
                level.notional = 0.0
                assigned_buy_levels += 1
                continue
            level.size = size
            level.notional = notional
            spent_budget += notional
            assigned_buy_levels += 1
        return grid

    def _buy_level_weights(self, regime: RegimeType, levels_count: int) -> list[float]:
        """Return normalized buy-budget weights for the current regime and buy ladder size."""
        if levels_count <= 0:
            return []
        if regime != RegimeType.UPTREND:
            equal_weight = 1.0 / levels_count
            return [equal_weight] * levels_count

        configured = list(self.config.grid.uptrend_buy_size_weights)
        if not configured:
            equal_weight = 1.0 / levels_count
            return [equal_weight] * levels_count

        if levels_count <= len(configured):
            raw_weights = configured[-levels_count:]
        else:
            raw_weights = configured + [configured[-1]] * (levels_count - len(configured))

        total_weight = sum(max(weight, 0.0) for weight in raw_weights)
        if total_weight <= 0:
            equal_weight = 1.0 / levels_count
            return [equal_weight] * levels_count

        normalized_weights = [max(weight, 0.0) / total_weight for weight in raw_weights]
        logger.info(
            "uptrend_buy_weights_applied levels=%s weights=%s",
            levels_count,
            ",".join(f"{weight:.4f}" for weight in normalized_weights),
        )
        return normalized_weights

    def _sell_level_weights(self, regime: RegimeType, levels_count: int, inventory: InventorySnapshot) -> list[float]:
        """Return normalized sell-size weights for the current regime and sell ladder size."""
        if levels_count <= 0:
            return []
        if regime != RegimeType.UPTREND:
            equal_weight = 1.0 / levels_count
            return [equal_weight] * levels_count

        configured = list(self.config.grid.uptrend_sell_size_weights)
        if not configured:
            equal_weight = 1.0 / levels_count
            return [equal_weight] * levels_count

        if levels_count <= len(configured):
            raw_weights = configured[-levels_count:]
        else:
            raw_weights = configured + [configured[-1]] * (levels_count - len(configured))

        total_weight = sum(max(weight, 0.0) for weight in raw_weights)
        if total_weight <= 0:
            equal_weight = 1.0 / levels_count
            return [equal_weight] * levels_count

        normalized_weights = [max(weight, 0.0) / total_weight for weight in raw_weights]
        normalized_weights = self._adaptive_sell_level_weights(normalized_weights, inventory)
        logger.info(
            "uptrend_sell_weights_applied levels=%s weights=%s",
            levels_count,
            ",".join(f"{weight:.4f}" for weight in normalized_weights),
        )
        return normalized_weights

    def _adaptive_sell_level_weights(self, base_weights: list[float], inventory: InventorySnapshot) -> list[float]:
        """Tilt higher sell levels upward when the inventory is materially in profit."""
        if not self.config.grid.adaptive_sell_sizing_enabled or len(base_weights) <= 1:
            return base_weights

        profit_ratio = self._unrealized_profit_ratio(inventory)
        if profit_ratio <= self.config.grid.adaptive_sell_sizing_profit_trigger_pct:
            return base_weights

        full_profit_pct = max(
            self.config.grid.adaptive_sell_sizing_full_profit_pct,
            self.config.grid.adaptive_sell_sizing_profit_trigger_pct + 1e-9,
        )
        progress = (
            (profit_ratio - self.config.grid.adaptive_sell_sizing_profit_trigger_pct)
            / (full_profit_pct - self.config.grid.adaptive_sell_sizing_profit_trigger_pct)
        )
        progress = max(min(progress, 1.0), 0.0)
        bias = self.config.grid.adaptive_sell_sizing_max_bias * progress
        center = (len(base_weights) - 1) / 2
        adjusted_weights = []
        for index, weight in enumerate(base_weights):
            if center <= 0:
                gradient = 0.0
            else:
                gradient = (index - center) / center
            adjusted_weights.append(max(weight * (1.0 + bias * gradient), 0.0))

        total_weight = sum(adjusted_weights)
        if total_weight <= 0:
            return base_weights
        return [weight / total_weight for weight in adjusted_weights]

    @staticmethod
    def _unrealized_profit_ratio(inventory: InventorySnapshot) -> float:
        """Return current unrealized profit ratio versus cost basis for active inventory."""
        if (
            inventory.base_balance <= 0
            or inventory.cost_basis_price is None
            or inventory.cost_basis_price <= 0
            or inventory.mark_price <= inventory.cost_basis_price
        ):
            return 0.0
        return (inventory.mark_price - inventory.cost_basis_price) / inventory.cost_basis_price

    def _viable_sell_level_count(
        self,
        sell_levels,
        inventory: InventorySnapshot,
        venue_constraints: VenueConstraints | None,
    ) -> int:
        """Return how many sell levels can be supported by current inventory on the venue."""
        total_levels = len(sell_levels)
        if total_levels <= 0 or inventory.base_balance <= 0:
            return 0
        if venue_constraints is None:
            return total_levels

        min_qty_requirement = max(venue_constraints.min_order_qty, 0.0)
        min_price = min((level.price for level in sell_levels if level.price > 0), default=0.0)
        min_notional_qty_requirement = 0.0
        effective_min_order_notional = self.config.execution.min_order_notional_usdt
        if venue_constraints.min_order_amt > 0:
            effective_min_order_notional = max(venue_constraints.min_order_amt, effective_min_order_notional)
        if effective_min_order_notional > 0 and min_price > 0:
            min_notional_qty_requirement = effective_min_order_notional / min_price

        required_qty_per_level = max(min_qty_requirement, min_notional_qty_requirement)
        if required_qty_per_level <= 0:
            return total_levels

        viable_levels = int(inventory.base_balance / required_qty_per_level)
        return max(min(total_levels, viable_levels), 0)

    def _viable_buy_level_count(
        self,
        buy_levels,
        effective_budget: float,
        venue_constraints: VenueConstraints | None,
    ) -> int:
        """Return how many buy levels can be funded while respecting the effective minimum order value."""
        total_levels = len(buy_levels)
        if total_levels <= 0 or effective_budget <= 0:
            return 0
        effective_min_order_notional = self._effective_min_order_notional(venue_constraints)
        if effective_min_order_notional <= 0:
            return total_levels
        viable_levels = int(effective_budget / effective_min_order_notional)
        return max(min(total_levels, viable_levels), 0)

    def _effective_min_order_notional(self, venue_constraints: VenueConstraints | None) -> float:
        """Return the minimum order value that planner-side level allocation should respect."""
        minimum = self.config.execution.min_order_notional_usdt
        if venue_constraints is not None and venue_constraints.min_order_amt > 0:
            minimum = max(minimum, venue_constraints.min_order_amt)
        return minimum

    def _normalize_size(self, size: float, venue_qty_step: float | None = None) -> float:
        """Round a size down to the configured execution step."""
        step = venue_qty_step if venue_qty_step is not None and venue_qty_step > 0 else self.config.execution.size_step
        if step <= 0:
            return size
        normalized = int(size / step) * step
        return round(normalized, 8)

    def _round_notional(self, notional: float) -> float:
        """Round order value to two decimal places to match exchange quote-amount precision."""
        return round(notional + 1e-9, 2)
