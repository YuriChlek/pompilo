"""Spot Grid infrastructure adapter package."""

from bot_platform_service.trading_bots.spot_grid.infrastructure.platform_snapshot_adapter import (
    DEFAULT_INDICATOR_REQUIRED_HISTORY,
    PlatformSnapshotIndicatorAdapter,
)

__all__ = [
    "DEFAULT_INDICATOR_REQUIRED_HISTORY",
    "PlatformSnapshotIndicatorAdapter",
]
