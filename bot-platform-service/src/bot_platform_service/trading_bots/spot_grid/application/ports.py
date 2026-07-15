from __future__ import annotations

from typing import Protocol

from bot_platform_service.domain import BotMarketSnapshot


class SpotGridSnapshotProvider(Protocol):
    """Port for loading platform market snapshots for Spot Grid planning."""

    async def get_snapshot(self, *, symbol: str, timeframe: str) -> BotMarketSnapshot:
        """Return a complete platform market snapshot for one symbol/timeframe."""
