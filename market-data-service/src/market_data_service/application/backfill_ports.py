from __future__ import annotations

from typing import Protocol

from market_data_service.domain.events.market_data_backfill_requested import MarketDataBackfillRequested


class BackfillRequestPort(Protocol):
    async def request_backfill(self, event: MarketDataBackfillRequested) -> bool: ...
