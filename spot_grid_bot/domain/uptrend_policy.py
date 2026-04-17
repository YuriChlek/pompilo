from __future__ import annotations

from dataclasses import dataclass, replace

from domain.grid_builder import round_to_step
from domain.market_models import IndicatorSnapshot, RegimeType


@dataclass(slots=True, frozen=True)
class UnderwaterRecoveryProfile:
    """Planner-side recovery posture for inventory below cost basis."""

    active: bool = False
    ratio: float = 0.0
    severity: str = "inactive"
    recovery_budget: float = 0.0
    max_buy_levels: int | None = None
    block_new_buys: bool = False
    sell_aggressiveness_atr: float = 0.0
    reason: str = ""


def is_uptrend_entry_extension_blocked(price: float, ema20: float, threshold_bps: float) -> bool:
    """Return whether price is too extended above ema20 to allow new uptrend entries."""
    if ema20 <= 0:
        return False
    extension_ratio = (price - ema20) / ema20
    return extension_ratio * 10_000 >= threshold_bps


def is_underwater_inventory(inventory) -> bool:
    """Return whether the current inventory mark price is below cost basis."""
    return (
        inventory.base_balance > 0
        and inventory.cost_basis_price is not None
        and inventory.cost_basis_price > 0
        and inventory.mark_price < inventory.cost_basis_price
    )


def underwater_ratio(inventory) -> float:
    """Return the current drawdown versus cost basis for the active inventory."""
    if not is_underwater_inventory(inventory):
        return 0.0
    return (inventory.cost_basis_price - inventory.mark_price) / inventory.cost_basis_price


def build_underwater_recovery_profile(inventory, regime: RegimeType, config) -> UnderwaterRecoveryProfile:
    """Return trigger-based underwater averaging posture for range/uptrend recovery only."""
    ratio = underwater_ratio(inventory)
    if not config.grid.underwater_averaging_enabled or ratio <= 0:
        return UnderwaterRecoveryProfile()
    if ratio < config.grid.underwater_averaging_trigger_pct:
        return UnderwaterRecoveryProfile(ratio=ratio, severity="below_trigger", reason="below_trigger")
    if ratio >= config.grid.underwater_deep_stop_pct:
        return UnderwaterRecoveryProfile(ratio=ratio, severity="deep_stop", block_new_buys=True, reason="deep_stop")
    if regime not in {RegimeType.RANGE, RegimeType.UPTREND}:
        return UnderwaterRecoveryProfile(ratio=ratio, severity="blocked_regime", block_new_buys=True, reason="blocked_regime")

    regime_budget_multiplier = (
        config.grid.underwater_uptrend_budget_multiplier
        if regime == RegimeType.UPTREND
        else config.grid.underwater_range_budget_multiplier
    )
    severity = "uptrend_recovery" if regime == RegimeType.UPTREND else "range_recovery"
    return UnderwaterRecoveryProfile(
        active=True,
        ratio=ratio,
        severity=severity,
        recovery_budget=config.grid.underwater_recovery_budget_pct * regime_budget_multiplier,
        max_buy_levels=max(1, config.grid.underwater_max_recovery_levels),
        sell_aggressiveness_atr=config.grid.underwater_recovery_sell_aggressiveness_atr * (
            0.75 if regime == RegimeType.UPTREND else 1.0
        ),
        reason=severity,
    )


def compute_uptrend_strength_score(price: float, indicators: IndicatorSnapshot, config) -> float:
    """Return a coarse [0, 1] score for current uptrend strength."""
    atr = max(indicators.atr14, 1e-9)
    price_anchor = max(price, 1e-9)
    ema_stack_score = 1.0 if indicators.ema20 > indicators.ema50 > indicators.ema200 else 0.0
    slope_score = min(
        max(indicators.ema50_slope, 0.0)
        / max(config.regime.ema_mid_slope_trend_threshold * 3.0, 1e-9),
        1.0,
    )
    ema_spread_score = min(max((indicators.ema20 - indicators.ema50) / price_anchor, 0.0) / 0.01, 1.0)
    price_confirmation = min(max((price - indicators.ema20) / atr, 0.0) / 1.5, 1.0)
    return max(min((ema_stack_score + slope_score + ema_spread_score + price_confirmation) / 4.0, 1.0), 0.0)


def adaptive_uptrend_take_profit_price(
    target_price: float,
    level_index: int,
    *,
    price: float,
    indicators: IndicatorSnapshot,
    config,
    tick_size: float | None = None,
    recovery_profile: UnderwaterRecoveryProfile | None = None,
) -> float:
    """Return an adaptive take-profit price that breathes with trend strength."""
    trend_strength = compute_uptrend_strength_score(price, indicators, config)
    atr = max(indicators.atr14, tick_size or 0.0, 1e-8)
    level_weight = 1.0 + max(level_index, 0) * 0.5
    atr_adjustment = 0.0

    if trend_strength >= config.grid.uptrend_strong_trend_threshold:
        atr_adjustment += config.grid.uptrend_adaptive_take_profit_bonus_atr * level_weight
    elif trend_strength <= config.grid.uptrend_weak_trend_threshold:
        atr_adjustment -= config.grid.uptrend_adaptive_take_profit_penalty_atr * (1.0 + max(level_index, 0) * 0.25)

    if recovery_profile is not None and recovery_profile.active:
        atr_adjustment -= recovery_profile.sell_aggressiveness_atr

    return max(round_to_step(target_price + atr * atr_adjustment, tick_size), tick_size or 0.0)


def limit_buy_levels(grid, max_buy_levels: int):
    """Return a grid with at most the requested number of buy levels preserved."""
    kept_buy_levels = 0
    limited_levels = []
    for level in grid.levels:
        if level.side.value == "BUY":
            kept_buy_levels += 1
            if kept_buy_levels > max_buy_levels:
                continue
        limited_levels.append(level)
    return replace(grid, levels=limited_levels)
