"""Machine-readable config schema for the platform-native Spot Greenwich skeleton."""

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
                    "default": "1d",
                    "allowed": ["1d", "4h"],
                },
                {
                    "key": "supporting_timeframes",
                    "type": "timeframe_list",
                    "label": "Supporting Timeframes",
                    "default": ["4h"],
                    "allowed": ["1d", "4h"],
                },
            ],
        },
        {
            "key": "signal_policy",
            "label": "Signal Policy",
            "description": "Fixture-safe controls for future Greenwich signal planning.",
            "fields": [
                {
                    "key": "min_confidence",
                    "type": "decimal",
                    "label": "Minimum Confidence",
                    "required": True,
                    "default": "0.50",
                    "min": "0.01",
                    "max": "1.00",
                },
                {
                    "key": "allow_hold_signals",
                    "type": "boolean",
                    "label": "Allow Hold Signals",
                    "default": True,
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

