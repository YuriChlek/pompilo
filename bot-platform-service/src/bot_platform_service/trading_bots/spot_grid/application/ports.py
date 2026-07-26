from __future__ import annotations

from typing import Protocol

from bot_platform_service.domain import BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.domain import PortfolioContext


class SpotGridSnapshotProvider(Protocol):
    """Port for loading platform market snapshots for Spot Grid planning."""

    async def get_snapshot(self, *, symbol: str, timeframe: str) -> BotMarketSnapshot:
        """Return a complete platform market snapshot for one symbol/timeframe."""


class SpotGridPortfolioContextProvider(Protocol):
    """Port for platform-supplied portfolio context used by Spot Grid planning."""

    async def get_portfolio_context(
        self,
        *,
        instance_id: str,
        symbol: str,
        timeframe: str,
        snapshot_id: str,
    ) -> PortfolioContext | None:
        """Return execution-neutral portfolio context or None for conservative fallback."""
