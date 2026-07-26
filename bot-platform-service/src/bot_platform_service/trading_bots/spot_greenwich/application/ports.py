from __future__ import annotations

from typing import Protocol

from bot_platform_service.domain import BotMarketSnapshot


class GreenwichSnapshotProvider(Protocol):
    """Application port for obtaining immutable platform market snapshots."""

    def get_snapshot(self, symbol: str, timeframe: str) -> BotMarketSnapshot:
        """Return one platform market snapshot for a symbol and timeframe."""

