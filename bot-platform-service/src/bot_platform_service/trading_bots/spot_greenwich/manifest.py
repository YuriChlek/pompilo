"""Lightweight metadata for the platform-native Spot Greenwich module."""

RAW_MANIFEST = {
    "module_id": "spot_greenwich",
    "display_name": "Spot Greenwich",
    "version": "0.1.0",
    "supported_modes": ("dry_run", "notification_only", "signal_only"),
    "required_timeframes": ("1d", "4h"),
    "required_market_data": ("snapshots",),
    "supports_multi_symbol": True,
    "config_schema_version": 1,
}

ADAPTER_PATH = "bot_platform_service.trading_bots.spot_greenwich.adapter"
ADAPTER_CLASS = "SpotGreenwichAdapter"

