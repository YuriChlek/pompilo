from __future__ import annotations

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

    def build_range_grid(self, price: float, indicators: IndicatorSnapshot, *, tick_size: float | None = None) -> GridSpec:
        """Build a symmetric range grid with buy and take-profit sell ladders."""
        width = indicators.atr14 * self.config.grid.range_grid_width_multiplier
        step = max(indicators.atr14 * self.config.grid.atr_grid_step_multiplier, tick_size or 0.0, 1e-8)
        range_low = price - width / 2
        range_high = price + width / 2
        mid = (range_low + range_high) / 2
        levels_count = max(self.config.grid.min_grid_levels, min(self.config.grid.max_active_levels, self.config.grid.max_grid_levels))
        lower_buy_band = range_low + (mid - range_low) * self.config.grid.range_buy_fraction
        upper_sell_band = mid + (range_high - mid) * (1 - self.config.grid.range_buy_fraction)
        buy_prices = self._ladder(lower_buy_band, range_low, levels_count, descending=True, tick_size=tick_size)
        sell_prices = self._ladder(upper_sell_band, range_high, levels_count, descending=False, tick_size=tick_size)
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
        configured_buy_levels = max(1, self.config.grid.uptrend_buy_levels)
        buy_multipliers = list(self.config.grid.uptrend_pullback_atr_multipliers[:configured_buy_levels])
        if not buy_multipliers:
            buy_multipliers = [self.config.grid.pullback_reentry_atr]

        configured_sell_levels = max(1, self.config.grid.uptrend_sell_levels)
        sell_multipliers = list(self.config.grid.uptrend_sell_atr_multipliers[:configured_sell_levels])
        if not sell_multipliers:
            sell_multipliers = [self.config.grid.rebalance_profit_target_atr]

        lower_bound = price - atr * max(buy_multipliers)
        upper_bound = price + atr * max(sell_multipliers)
        buy_prices = [round_to_step(price - atr * multiplier, tick_size) for multiplier in buy_multipliers]
        sell_prices = [round_to_step(price + atr * multiplier, tick_size) for multiplier in sell_multipliers]
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
        step = min(step_candidates) if step_candidates else atr * min(buy_multipliers)
        return GridSpec(RegimeType.UPTREND, price, lower_bound, upper_bound, max(step, tick_size or 0.0, 1e-8), levels)

    def _ladder(self, start: float, end: float, count: int, descending: bool, *, tick_size: float | None = None) -> list[float]:
        """Generate a stepped price ladder between two anchors."""
        if count <= 1:
            return [max(tick_size or 0.0, round_to_step(start, tick_size))]
        delta = abs(start - end) / (count - 1)
        values = [start - delta * index if descending else start + delta * index for index in range(count)]
        minimum_price = tick_size or 0.0
        return [max(minimum_price, round_to_step(value, tick_size)) for value in values]
