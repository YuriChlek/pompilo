from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from market_data_service.domain.enums import MarketDataSource


@dataclass(frozen=True, slots=True)
class CandleCollectionState:
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    timeframe: str
    bootstrap_from: datetime
    bootstrap_to: datetime
    bootstrap_next_from: datetime
    bootstrap_completed_at: datetime | None
    last_successful_close_time: datetime | None

    @property
    def bootstrap_completed(self) -> bool:
        return self.bootstrap_completed_at is not None
