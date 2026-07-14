from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from market_data_service.domain.enums import MarketDataSource, MarketSymbolStatus, ProviderSymbolStatus


@dataclass(frozen=True, slots=True)
class MarketSymbol:
    canonical_symbol: str
    base_asset: str
    quote_asset: str
    status: MarketSymbolStatus = MarketSymbolStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class ProviderSymbol:
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    status: ProviderSymbolStatus
    supported_timeframes: tuple[str, ...]
    max_backfill_days: int | None = None
    metadata: Mapping[str, object] | None = None

    def supports_all_timeframes(self, required_timeframes: tuple[str, ...]) -> bool:
        supported = set(self.supported_timeframes)
        return all(timeframe in supported for timeframe in required_timeframes)

    @property
    def is_trading(self) -> bool:
        return self.status == ProviderSymbolStatus.TRADING
