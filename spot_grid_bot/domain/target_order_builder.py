from __future__ import annotations

import logging

from domain.market_structure import StructureSnapshot
from domain.allocation import calculate_symbol_entry_budget
from domain.cost_basis import minimum_exit_price, rebased_take_profit_price
from domain.de_risk import build_de_risk_orders
from domain.grid_viability import apply_venue_viability
from domain.models import IndicatorSnapshot, OrderSide, RegimeType, RiskDecision, TargetOrder
from domain.range_entry_policy import evaluate_range_entry_profile
from domain.uptrend_policy import (
    adaptive_uptrend_take_profit_price,
    build_underwater_recovery_profile,
    is_uptrend_entry_extension_blocked,
    limit_buy_levels,
)

logger = logging.getLogger(__name__)


def build_target_orders(
    *,
    price: float,
    indicators: IndicatorSnapshot,
    inventory,
    regime: RegimeType,
    risk: RiskDecision,
    symbol: str,
    config,
    grid_builder,
    inventory_manager,
    tick_size: float | None = None,
    venue_constraints=None,
    portfolio_budget: float | None = None,
    recovery_budget: float | None = None,
    bars_in_state: int = 0,
    structure_snapshot: StructureSnapshot | None = None,
) -> list[TargetOrder]:
    """Build the next target order set for the current regime and risk state."""
    if risk.force_risk_off or regime == RegimeType.HIGH_VOLATILITY:
        return _build_de_risk_orders(inventory, price, symbol, risk, config)
    if regime == RegimeType.RANGE:
        grid = grid_builder.build_range_grid(
            price,
            indicators,
            tick_size=tick_size,
            structure_snapshot=structure_snapshot,
        )
    elif regime == RegimeType.UPTREND:
        grid = grid_builder.build_trend_pullback_grid(price, indicators, tick_size=tick_size)
    elif regime == RegimeType.DOWNTREND:
        return _build_de_risk_orders(inventory, price, symbol, risk, config)
    else:
        return []

    symbol_entry_budget = calculate_symbol_entry_budget(
        inventory,
        risk,
        config,
        portfolio_budget=portfolio_budget,
    )
    recovery_profile = build_underwater_recovery_profile(
        inventory,
        regime,
        config,
        bars_in_state=bars_in_state,
    )
    is_recovery_averaging_active = False
    if recovery_profile.active:
        original_budget = symbol_entry_budget
        local_recovery_budget = inventory.available_quote * recovery_profile.recovery_budget
        portfolio_recovery_budget = float("inf") if recovery_budget is None else max(recovery_budget, 0.0)
        remaining_inventory_room = min(
            max(inventory.total_equity * config.risk.max_symbol_inventory_pct_of_equity - inventory.inventory_notional, 0.0),
            max(config.risk.max_symbol_notional_cap - inventory.inventory_notional, 0.0),
            max(config.risk.max_inventory_notional - inventory.inventory_notional, 0.0),
        )
        symbol_entry_budget = min(
            local_recovery_budget,
            portfolio_recovery_budget,
            remaining_inventory_room,
            inventory.available_quote,
        )
        if recovery_profile.max_buy_levels is not None:
            grid = limit_buy_levels(grid, recovery_profile.max_buy_levels)
        is_recovery_averaging_active = True
        logger.info(
            "underwater_averaging_activated symbol=%s regime=%s profile=%s underwater_ratio=%.4f original_symbol_budget=%.4f free_quote=%.4f recovery_budget=%.4f max_buy_levels=%s cost_basis=%s mark_price=%s",
            symbol.upper(),
            regime.value,
            recovery_profile.severity,
            recovery_profile.ratio,
            original_budget,
            inventory.available_quote,
            symbol_entry_budget,
            recovery_profile.max_buy_levels,
            inventory.cost_basis_price,
            inventory.mark_price,
        )
    elif recovery_profile.reason:
        logger.info(
            "underwater_averaging_blocked symbol=%s regime=%s reason=%s underwater_ratio=%.4f cost_basis=%s mark_price=%s",
            symbol.upper(),
            regime.value,
            recovery_profile.reason,
            recovery_profile.ratio,
            inventory.cost_basis_price,
            inventory.mark_price,
        )

    block_range_entries = False
    if regime == RegimeType.RANGE and not is_recovery_averaging_active:
        range_entry_profile = evaluate_range_entry_profile(
            price=price,
            indicators=indicators,
            grid=grid,
            config=config,
        )
        if range_entry_profile.budget_penalty < 1.0:
            symbol_entry_budget *= range_entry_profile.budget_penalty
        if range_entry_profile.max_buy_levels is not None:
            grid = limit_buy_levels(grid, range_entry_profile.max_buy_levels)
        block_range_entries = range_entry_profile.block_new_buys
        if (
            range_entry_profile.budget_penalty < 1.0
            or range_entry_profile.max_buy_levels is not None
            or block_range_entries
        ):
            logger.info(
                "range_entry_quality_applied symbol=%s quality_score=%.4f budget_penalty=%.2f max_buy_levels=%s block_new_buys=%s reasons=%s",
                symbol.upper(),
                range_entry_profile.quality_score,
                range_entry_profile.budget_penalty,
                range_entry_profile.max_buy_levels,
                block_range_entries,
                ",".join(range_entry_profile.reasons),
            )
    grid = inventory_manager.allocate_grid(
        grid,
        inventory,
        symbol_entry_budget,
        venue_constraints=venue_constraints,
    )
    target_orders: list[TargetOrder] = []
    blocked_no_loss_sell_count = 0
    block_uptrend_entries = regime == RegimeType.UPTREND and is_uptrend_entry_extension_blocked(
        price,
        indicators.ema20,
        config.grid.uptrend_max_price_extension_from_ema20_bps,
    )
    if block_uptrend_entries:
        extension_bps = ((price - indicators.ema20) / indicators.ema20) * 10_000 if indicators.ema20 > 0 else 0.0
        logger.info(
            "uptrend_entry_blocked_due_to_extension symbol=%s price=%s ema20=%s extension_bps=%.2f threshold_bps=%.2f",
            symbol.upper(),
            price,
            indicators.ema20,
            extension_bps,
            config.grid.uptrend_max_price_extension_from_ema20_bps,
        )
    for level in grid.levels:
        if level.size <= 0:
            continue
        if ((risk.pause_entries or risk.cancel_entries) and level.side == OrderSide.BUY) or (
            block_uptrend_entries and level.side == OrderSide.BUY
        ) or (
            block_range_entries and level.side == OrderSide.BUY
        ) or (
            recovery_profile.block_new_buys and level.side == OrderSide.BUY
        ):
            continue
        target_price = level.price
        if level.side == OrderSide.SELL:
            if regime == RegimeType.UPTREND:
                target_price = adaptive_uptrend_take_profit_price(
                    target_price,
                    level.level_index,
                    price=price,
                    indicators=indicators,
                    config=config,
                    tick_size=tick_size,
                    recovery_profile=recovery_profile,
                )
            target_price = rebased_take_profit_price(
                target_price=target_price,
                level_index=level.level_index,
                inventory=inventory,
                indicators=indicators,
                strategy_config=config,
                tick_size=tick_size,
            )
        if level.side == OrderSide.SELL and not _is_sell_order_allowed(inventory, target_price, config):
            blocked_no_loss_sell_count += 1
            continue
        target_orders.append(
            TargetOrder(
                client_order_id=f"{regime.value.lower()}-{level.tag}-{level.level_index}",
                symbol=symbol,
                side=level.side,
                price=target_price,
                size=level.size,
                reduce_only=level.side == OrderSide.SELL,
                tag=level.tag,
            )
        )
    if blocked_no_loss_sell_count > 0:
        logger.warning(
            "planner_no_loss_sell_blocked symbol=%s blocked_levels=%s min_exit_price=%s mark_price=%s cost_basis=%s",
            symbol.upper(),
            blocked_no_loss_sell_count,
            minimum_exit_price(inventory, config),
            inventory.mark_price,
            inventory.cost_basis_price,
        )
    viable_target_orders = apply_venue_viability(target_orders, venue_constraints)
    if venue_constraints is not None and len(viable_target_orders) != len(target_orders):
        raw_buy_count = sum(1 for order in target_orders if order.side == OrderSide.BUY)
        raw_sell_count = sum(1 for order in target_orders if order.side == OrderSide.SELL)
        viable_buy_count = sum(1 for order in viable_target_orders if order.side == OrderSide.BUY)
        viable_sell_count = sum(1 for order in viable_target_orders if order.side == OrderSide.SELL)
        logger.info(
            "planner_grid_viability_applied symbol=%s raw_orders=%s viable_orders=%s raw_buy_levels=%s viable_buy_levels=%s raw_sell_levels=%s viable_sell_levels=%s tick_size=%s",
            symbol.upper(),
            len(target_orders),
            len(viable_target_orders),
            raw_buy_count,
            viable_buy_count,
            raw_sell_count,
            viable_sell_count,
            venue_constraints.tick_size,
        )
    return viable_target_orders


def _build_de_risk_orders(inventory, price: float, symbol: str, risk: RiskDecision, config) -> list[TargetOrder]:
    """Delegate staged de-risk order generation for protective regimes."""
    return build_de_risk_orders(
        symbol=symbol,
        inventory=inventory,
        live_price=price,
        strategy_config=config,
        mode=risk.de_risk_mode,
    )


def _is_sell_order_allowed(inventory, target_price: float, config) -> bool:
    """Check whether a sell target satisfies the current no-loss exit floor."""
    if inventory.base_balance <= 0:
        return False
    min_allowed_price = minimum_exit_price(inventory, config)
    if min_allowed_price is None:
        return False
    return target_price >= min_allowed_price
