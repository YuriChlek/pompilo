"""Lightweight metadata for the platform-native Spot Grid module."""

RAW_MANIFEST = {
    "module_id": "spot_grid",
    "display_name": "Spot Grid",
    "version": "0.1.0",
    "supported_modes": ("dry_run", "notification_only", "signal_only"),
    "required_timeframes": ("1h", "4h"),
    "required_market_data": ("snapshots",),
    "supports_multi_symbol": True,
    "config_schema_version": 1,
}

ADAPTER_PATH = "bot_platform_service.trading_bots.spot_grid.adapter"
ADAPTER_CLASS = "SpotGridAdapter"

