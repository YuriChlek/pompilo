from __future__ import annotations

from dataclasses import dataclass

from domain.market_models import IndicatorSnapshot
from domain.order_models import GridSpec


@dataclass(slots=True, frozen=True)
class RangeEntryProfile:
    """Planner-side range entry tuning based on current range quality."""

    quality_score: float
    budget_penalty: float = 1.0
    max_buy_levels: int | None = None
    block_new_buys: bool = False
    reasons: tuple[str, ...] = ()


def evaluate_range_entry_profile(
    *,
    price: float,
    indicators: IndicatorSnapshot,
    grid: GridSpec,
    config,
) -> RangeEntryProfile:
    """Return whether range entries should stay normal, soften, or pause entirely."""
    atr = max(indicators.atr14, 1e-9)
    atr_units_width = indicators.range_width / atr
    reasons: list[str] = []

    slope_pressure = min(
        abs(indicators.ema50_slope) / max(config.regime.ema_mid_slope_flat_threshold * 3.0, 1e-9),
        2.0,
    )
    direction_pressure = min(
        indicators.directional_move / max(config.regime.range_directional_threshold * 1.5, 1e-9),
        2.0,
    )
    upper_band_pressure = min(max(indicators.price_vs_ema50, 0.0) / 0.02, 1.0)
    width_penalty = 0.0
    if atr_units_width < config.regime.range_width_atr_min:
        width_penalty = min(
            (config.regime.range_width_atr_min - atr_units_width)
            / max(config.regime.range_width_atr_min, 1e-9),
            1.0,
        )
    elif atr_units_width > config.regime.range_width_atr_max:
        width_penalty = min(
            (atr_units_width - config.regime.range_width_atr_max)
            / max(config.regime.range_width_atr_max, 1e-9),
            1.0,
        )

    quality_score = 1.0 - (
        0.20 * (slope_pressure / 2.0)
        + 0.20 * (direction_pressure / 2.0)
        + 0.15 * min(upper_band_pressure, 1.0)
        + 0.10 * width_penalty
    )

    if config.grid.range_rsi_filter_enabled:
        if indicators.rsi14 >= config.grid.range_rsi_overbought_threshold:
            quality_score -= config.grid.range_rsi_overbought_penalty
            reasons.append("range_entry_rsi_overbought")
        elif indicators.rsi14 <= config.grid.range_rsi_oversold_threshold:
            quality_score += config.grid.range_rsi_oversold_bonus
            reasons.append("range_entry_rsi_oversold")

    quality_score = max(min(quality_score, 1.0), 0.0)

    breakdown_risk = (
        price <= indicators.ema20
        and indicators.ema20 <= indicators.ema50
        and indicators.ema50_slope < 0
        and indicators.directional_sign < 0
        and indicators.directional_move
        >= config.regime.range_directional_threshold * config.grid.range_breakdown_directional_threshold_multiplier
    )
    if breakdown_risk:
        return RangeEntryProfile(
            quality_score=quality_score,
            block_new_buys=True,
            reasons=tuple([*reasons, "range_breakdown_risk"]),
        )

    if quality_score <= config.grid.range_entry_quality_hard_threshold:
        return RangeEntryProfile(
            quality_score=quality_score,
            budget_penalty=config.grid.range_poor_entry_budget_penalty,
            max_buy_levels=max(1, config.grid.range_poor_max_buy_levels),
            reasons=tuple([*reasons, "range_entry_quality_poor"]),
        )

    if quality_score <= config.grid.range_entry_quality_soft_threshold:
        return RangeEntryProfile(
            quality_score=quality_score,
            budget_penalty=config.grid.range_weak_entry_budget_penalty,
            max_buy_levels=max(1, config.grid.range_weak_max_buy_levels),
            reasons=tuple([*reasons, "range_entry_quality_weak"]),
        )

    return RangeEntryProfile(quality_score=quality_score, reasons=tuple(reasons))
