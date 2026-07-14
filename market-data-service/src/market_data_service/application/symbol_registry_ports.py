from __future__ import annotations

from typing import Protocol

from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.symbol_registry_models import ProviderSymbol


class ProviderSymbolRegistryPort(Protocol):
    async def get_provider_symbol(self, source: MarketDataSource, provider_symbol: str) -> ProviderSymbol | None: ...
