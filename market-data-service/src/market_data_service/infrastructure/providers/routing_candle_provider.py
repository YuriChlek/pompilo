from __future__ import annotations

from datetime import datetime
from typing import Mapping

from market_data_service.application.market_data_ports import CandleProviderPort
from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.symbol_registry_models import ProviderSymbol


class RoutingCandleProvider:
    """Delegates candle fetching to the appropriate provider adapter based on symbol source."""

    def __init__(self, adapters: Mapping[MarketDataSource, CandleProviderPort]) -> None:
        self.adapters = adapters

    async def fetch_closed_candles(
        self,
        provider_symbol: ProviderSymbol,
        *,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[CanonicalCandle]:
        adapter = self.adapters.get(provider_symbol.source)
        if adapter is None:
            raise ValueError(f"No adapter registered for source: {provider_symbol.source}")
        return await adapter.fetch_closed_candles(
            provider_symbol,
            timeframe=timeframe,
            from_time=from_time,
            to_time=to_time,
        )
