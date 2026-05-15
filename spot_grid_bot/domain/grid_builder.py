from __future__ import annotations

from domain.market_structure import StructureSnapshot
from domain.models import GridLevel, GridSpec, IndicatorSnapshot, OrderSide, RegimeType
from domain.strategy_config import StrategyConfig


def round_to_step(value: float, step: float | None) -> float:
    """Round a price-like value to the requested symbol step when available."""
    if step is None or step <= 0:
        return round(value, 8)
    return round(value / step) * step


class GridBuilder:
    """Build regime-specific grid geometry before inventory sizing is applied."""

    def __init__(self, config: StrategyConfig) -> None:
        """Store strategy configuration used for grid construction."""
        self.config = config

    def build_range_grid(
        self,
        price: float,
        indicators: IndicatorSnapshot,
        *,
        tick_size: float | None = None,
        structure_snapshot: StructureSnapshot | None = None,
    ) -> GridSpec:
        """Build a range grid with price-position-aware buy and sell ladders."""
        volatility_multiplier = self._volatility_step_multiplier(indicators)
        width = indicators.atr14 * self.config.grid.range_grid_width_multiplier * volatility_multiplier
        step = max(
            indicators.atr14 * self.config.grid.atr_grid_step_multiplier * volatility_multiplier,
            tick_size or 0.0,
            1e-8,
        )
        range_low = price - width / 2
        range_high = price + width / 2
        mid = (range_low + range_high) / 2
        levels_count = max(self.config.grid.min_grid_levels, min(self.config.grid.max_active_levels, self.config.grid.max_grid_levels))
        buy_levels_count, sell_levels_count = self._range_level_counts(indicators.range_position, levels_count)
        lower_buy_band = range_low + (mid - range_low) * self.config.grid.range_buy_fraction
        upper_sell_band = mid + (range_high - mid) * (1 - self.config.grid.range_buy_fraction)
        buy_prices = self._ladder(lower_buy_band, range_low, buy_levels_count, descending=True, tick_size=tick_size)
        sell_prices = self._ladder(upper_sell_band, range_high, sell_levels_count, descending=False, tick_size=tick_size)
        buy_prices = self._align_range_prices_to_structure(
            base_prices=buy_prices,
            swing_prices=[] if structure_snapshot is None else [point.price for point in structure_snapshot.swing_lows],
            lower_bound=range_low,
            upper_bound=mid,
            descending=True,
            tick_size=tick_size,
        )
        sell_prices = self._align_range_prices_to_structure(
            base_prices=sell_prices,
            swing_prices=[] if structure_snapshot is None else [point.price for point in structure_snapshot.swing_highs],
            lower_bound=mid,
            upper_bound=range_high,
            descending=False,
            tick_size=tick_size,
        )
        levels = [
            GridLevel(price=level_price, size=0.0, side=OrderSide.BUY, level_index=index, notional=0.0, tag="range_buy")
            for index, level_price in enumerate(buy_prices)
        ]
        levels.extend(
            GridLevel(price=level_price, size=0.0, side=OrderSide.SELL, level_index=index, notional=0.0, tag="range_take_profit")
            for index, level_price in enumerate(sell_prices)
        )
        return GridSpec(RegimeType.RANGE, price, range_low, range_high, step, levels)

    def build_trend_pullback_grid(self, price: float, indicators: IndicatorSnapshot, *, tick_size: float | None = None) -> GridSpec:
        """Build an ATR pullback ladder for uptrend continuation with wider take-profit sells."""
        atr = max(indicators.atr14, tick_size or 0.0, 1e-8)
        effective_atr = atr * self._volatility_step_multiplier(indicators)
        configured_buy_levels = max(1, self.config.grid.uptrend_buy_levels)
        buy_multipliers = list(self.config.grid.uptrend_pullback_atr_multipliers[:configured_buy_levels])
        if not buy_multipliers:
            buy_multipliers = [self.config.grid.pullback_reentry_atr]

        configured_sell_levels = max(1, self.config.grid.uptrend_sell_levels)
        sell_multipliers = list(self.config.grid.uptrend_sell_atr_multipliers[:configured_sell_levels])
        if not sell_multipliers:
            sell_multipliers = [self.config.grid.rebalance_profit_target_atr]

        lower_bound = price - effective_atr * max(buy_multipliers)
        upper_bound = price + effective_atr * max(sell_multipliers)
        buy_prices = [round_to_step(price - effective_atr * multiplier, tick_size) for multiplier in buy_multipliers]
        sell_prices = [round_to_step(price + effective_atr * multiplier, tick_size) for multiplier in sell_multipliers]
        levels = [
            GridLevel(price=level_price, size=0.0, side=OrderSide.BUY, level_index=index, notional=0.0, tag="trend_pullback_buy")
            for index, level_price in enumerate(buy_prices)
        ]
        levels.extend(
            GridLevel(price=level_price, size=0.0, side=OrderSide.SELL, level_index=index, notional=0.0, tag="trend_recovery_take_profit")
            for index, level_price in enumerate(sell_prices)
        )
        step_candidates = [
            abs(buy_prices[index] - buy_prices[index + 1])
            for index in range(len(buy_prices) - 1)
            if abs(buy_prices[index] - buy_prices[index + 1]) > 0
        ]
        step = min(step_candidates) if step_candidates else effective_atr * min(buy_multipliers)
        return GridSpec(RegimeType.UPTREND, price, lower_bound, upper_bound, max(step, tick_size or 0.0, 1e-8), levels)

    def _ladder(self, start: float, end: float, count: int, descending: bool, *, tick_size: float | None = None) -> list[float]:
        """Generate a stepped price ladder between two anchors."""
        if count <= 1:
            return [max(tick_size or 0.0, round_to_step(start, tick_size))]
        delta = abs(start - end) / (count - 1)
        values = [start - delta * index if descending else start + delta * index for index in range(count)]
        minimum_price = tick_size or 0.0
        return [max(minimum_price, round_to_step(value, tick_size)) for value in values]

    def _range_level_counts(self, range_position: float, base_levels_count: int) -> tuple[int, int]:
        """Return buy/sell level counts based on the current position inside the recent range."""
        if not self.config.grid.range_asymmetry_enabled or base_levels_count <= 1:
            return base_levels_count, base_levels_count

        shift = self._range_level_shift(range_position)
        buy_levels_count = max(1, base_levels_count - shift)
        sell_levels_count = max(1, base_levels_count + shift)
        return buy_levels_count, sell_levels_count

    def _range_level_shift(self, range_position: float) -> int:
        """Return how strongly the range grid should lean toward buys or sells."""
        capped_position = max(min(range_position, 1.0), 0.0)
        soft_threshold = self.config.grid.range_asymmetry_soft_position_threshold
        hard_threshold = self.config.grid.range_asymmetry_hard_position_threshold
        max_shift = max(self.config.grid.range_asymmetry_max_level_shift, 0)
        if max_shift <= 0:
            return 0

        if capped_position >= hard_threshold:
            return max_shift
        if capped_position >= soft_threshold:
            return min(1, max_shift)
        mirrored_position = 1.0 - capped_position
        if mirrored_position >= hard_threshold:
            return -max_shift
        if mirrored_position >= soft_threshold:
            return -min(1, max_shift)
        return 0

    def _volatility_step_multiplier(self, indicators: IndicatorSnapshot) -> float:
        """Return a clamped step multiplier from short-vs-long realized volatility."""
        if not self.config.grid.adaptive_grid_step_enabled:
            return 1.0
        raw_ratio = indicators.volatility_regime_ratio
        if raw_ratio <= 0:
            return 1.0
        return max(
            self.config.grid.adaptive_grid_step_ratio_min,
            min(self.config.grid.adaptive_grid_step_ratio_max, raw_ratio),
        )

    def _align_range_prices_to_structure(
        self,
        *,
        base_prices: list[float],
        swing_prices: list[float],
        lower_bound: float,
        upper_bound: float,
        descending: bool,
        tick_size: float | None = None,
    ) -> list[float]:
        """Shift part of the range ladder toward nearby swing support or resistance levels."""
        if not self.config.grid.range_structure_alignment_enabled or not base_prices or not swing_prices:
            return base_prices

        candidate_prices = [
            round_to_step(price, tick_size)
            for price in swing_prices
            if lower_bound <= price <= upper_bound
        ]
        if not candidate_prices:
            return base_prices

        candidate_prices = sorted(set(candidate_prices), reverse=descending)
        max_anchored_levels = min(self.config.grid.range_structure_max_anchored_levels, len(base_prices), len(candidate_prices))
        if max_anchored_levels <= 0:
            return base_prices

        aligned_prices = list(base_prices)
        used_indices: set[int] = set()
        for anchor_price in candidate_prices[:max_anchored_levels]:
            nearest_index = self._nearest_price_index(aligned_prices, anchor_price, used_indices)
            if nearest_index is None:
                continue
            aligned_prices[nearest_index] = anchor_price
            used_indices.add(nearest_index)

        ordered_prices = sorted(aligned_prices, reverse=descending)
        unique_prices: list[float] = []
        for price in ordered_prices:
            if price not in unique_prices:
                unique_prices.append(price)
        if len(unique_prices) >= len(base_prices):
            return unique_prices[: len(base_prices)]

        for price in sorted(base_prices, reverse=descending):
            if price in unique_prices:
                continue
            unique_prices.append(price)
            if len(unique_prices) >= len(base_prices):
                break
        return unique_prices

    @staticmethod
    def _nearest_price_index(prices: list[float], anchor_price: float, used_indices: set[int]) -> int | None:
        """Return the closest unused ladder index for a structure anchor."""
        nearest_index: int | None = None
        nearest_distance: float | None = None
        for index, price in enumerate(prices):
            if index in used_indices:
                continue
            distance = abs(price - anchor_price)
            if nearest_distance is None or distance < nearest_distance:
                nearest_index = index
                nearest_distance = distance
        return nearest_index
