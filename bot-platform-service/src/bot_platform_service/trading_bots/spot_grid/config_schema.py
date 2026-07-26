"""Machine-readable config schema for platform-native Spot Grid planning."""

from bot_platform_service.trading_bots.spot_grid.bot_config import (
    DEFAULT_SPOT_GRID_CONFIG,
    HIGH_VOLATILITY_PAUSE_THRESHOLD_BOUNDS,
    MAX_GRID_LEVELS_BOUNDS,
    MAX_POSITION_FRACTION_BOUNDS,
    MIN_PRICE_DISTANCE_FRACTION_BOUNDS,
    SUPPORTED_TIMEFRAMES,
    VOLATILITY_COOLDOWN_RUNS_BOUNDS,
)

CONFIG_SCHEMA = {
    "schema_version": 1,
    "sections": [
        {
            "key": "market_data",
            "label": "Market Data",
            "description": "Symbols and timeframes consumed from platform snapshots.",
            "fields": [
                {
                    "key": "symbols",
                    "type": "symbol_list",
                    "label": "Symbols",
                    "required": True,
                    "default": list(DEFAULT_SPOT_GRID_CONFIG.symbols),
                },
                {
                    "key": "primary_timeframe",
                    "type": "timeframe",
                    "label": "Primary Timeframe",
                    "required": True,
                    "default": DEFAULT_SPOT_GRID_CONFIG.primary_timeframe,
                    "allowed": list(SUPPORTED_TIMEFRAMES),
                },
                {
                    "key": "supporting_timeframes",
                    "type": "timeframe_list",
                    "label": "Supporting Timeframes",
                    "default": list(DEFAULT_SPOT_GRID_CONFIG.supporting_timeframes),
                    "allowed": list(SUPPORTED_TIMEFRAMES),
                },
            ],
        },
        {
            "key": "risk",
            "label": "Risk",
            "description": "Signal-only risk limits for Spot Grid target intents.",
            "fields": [
                {
                    "key": "max_position_fraction",
                    "type": "decimal",
                    "label": "Max Position Fraction",
                    "required": True,
                    "default": str(DEFAULT_SPOT_GRID_CONFIG.max_position_fraction),
                    "min": str(MAX_POSITION_FRACTION_BOUNDS.min),
                    "max": str(MAX_POSITION_FRACTION_BOUNDS.max),
                },
                {
                    "key": "max_grid_levels",
                    "type": "integer",
                    "label": "Max Grid Levels",
                    "required": True,
                    "default": DEFAULT_SPOT_GRID_CONFIG.max_grid_levels,
                    "min": MAX_GRID_LEVELS_BOUNDS.min,
                    "max": MAX_GRID_LEVELS_BOUNDS.max,
                },
                {
                    "key": "min_price_distance_fraction",
                    "type": "decimal",
                    "label": "Min Price Distance Fraction",
                    "required": True,
                    "default": str(DEFAULT_SPOT_GRID_CONFIG.min_price_distance_fraction),
                    "min": str(MIN_PRICE_DISTANCE_FRACTION_BOUNDS.min),
                    "max": str(MIN_PRICE_DISTANCE_FRACTION_BOUNDS.max),
                },
                {
                    "key": "high_volatility_pause_threshold",
                    "type": "decimal",
                    "label": "High Volatility Pause Threshold",
                    "required": True,
                    "default": str(DEFAULT_SPOT_GRID_CONFIG.high_volatility_pause_threshold),
                    "min": str(HIGH_VOLATILITY_PAUSE_THRESHOLD_BOUNDS.min),
                    "max": str(HIGH_VOLATILITY_PAUSE_THRESHOLD_BOUNDS.max),
                },
                {
                    "key": "volatility_cooldown_runs",
                    "type": "integer",
                    "label": "Volatility Cooldown Runs",
                    "required": True,
                    "default": DEFAULT_SPOT_GRID_CONFIG.volatility_cooldown_runs,
                    "min": VOLATILITY_COOLDOWN_RUNS_BOUNDS.min,
                    "max": VOLATILITY_COOLDOWN_RUNS_BOUNDS.max,
                },
            ],
        },
        {
            "key": "runtime",
            "label": "Runtime",
            "description": "Signal-only runtime controls.",
            "fields": [
                {
                    "key": "emit_diagnostics",
                    "type": "boolean",
                    "label": "Emit Diagnostics",
                    "default": DEFAULT_SPOT_GRID_CONFIG.emit_diagnostics,
                }
            ],
        },
    ],
}
