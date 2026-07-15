from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    GridLevel,
    GridLevelSide,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlan,
)

DEFAULT_PRICE_QUANT = Decimal("0.00000001")


class SpotGridPlanner:
    """Pure Decimal-based planner for platform-native Spot Grid runs."""

    def plan(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: tuple[SpotGridCandle, ...],
        config: SpotGridConfig,
    ) -> SpotGridPlan:
        """Build a balanced buy/sell grid from immutable platform candles."""
        if not candles:
            raise ValueError("candles must not be empty")
        if config.max_grid_levels <= 0:
            raise ValueError("max_grid_levels must be positive")

        reference_price = candles[-1].close
        range_low = min(candle.low for candle in candles)
        range_high = max(candle.high for candle in candles)
        if range_high <= range_low:
            band = max(reference_price * Decimal("0.01"), DEFAULT_PRICE_QUANT)
            range_low = reference_price - band
            range_high = reference_price + band

        step = (range_high - range_low) / Decimal(config.max_grid_levels + 1)
        levels = []
        for index in range(config.max_grid_levels):
            level_offset = Decimal(index + 1) * step
            buy_price = _quantize_price(reference_price - level_offset)
            sell_price = _quantize_price(reference_price + level_offset)
            if buy_price > Decimal("0"):
                levels.append(
                    GridLevel(
                        side=GridLevelSide.BUY,
                        price=buy_price,
                        level_index=index,
                        reason="range_buy",
                    )
                )
            levels.append(
                GridLevel(
                    side=GridLevelSide.SELL,
                    price=sell_price,
                    level_index=index,
                    reason="range_take_profit",
                )
            )

        return SpotGridPlan(
            symbol=symbol.upper(),
            timeframe=timeframe,
            reference_price=_quantize_price(reference_price),
            range_low=_quantize_price(range_low),
            range_high=_quantize_price(range_high),
            levels=tuple(levels),
            diagnostics={
                "planner": "spot_grid_decimal_grid",
                "candle_count": len(candles),
                "max_grid_levels": config.max_grid_levels,
                "max_position_fraction": str(config.max_position_fraction),
            },
        )


def _quantize_price(value: Decimal) -> Decimal:
    return value.quantize(DEFAULT_PRICE_QUANT, rounding=ROUND_HALF_UP)
