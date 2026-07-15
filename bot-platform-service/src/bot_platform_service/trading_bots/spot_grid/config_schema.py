"""Machine-readable config schema for the platform-native Spot Grid skeleton."""

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
                    "default": ["ETHUSDT"],
                },
                {
                    "key": "primary_timeframe",
                    "type": "timeframe",
                    "label": "Primary Timeframe",
                    "required": True,
                    "default": "1h",
                    "allowed": ["1h", "4h"],
                },
                {
                    "key": "supporting_timeframes",
                    "type": "timeframe_list",
                    "label": "Supporting Timeframes",
                    "default": ["4h"],
                    "allowed": ["1h", "4h"],
                },
            ],
        },
        {
            "key": "risk",
            "label": "Risk",
            "description": "Fixture-safe risk limits for future grid planning.",
            "fields": [
                {
                    "key": "max_position_fraction",
                    "type": "decimal",
                    "label": "Max Position Fraction",
                    "required": True,
                    "default": "0.10",
                    "min": "0.01",
                    "max": "1.00",
                },
                {
                    "key": "max_grid_levels",
                    "type": "integer",
                    "label": "Max Grid Levels",
                    "required": True,
                    "default": 6,
                    "min": 1,
                    "max": 50,
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
                    "default": True,
                }
            ],
        },
    ],
}

