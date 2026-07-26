from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.persistence.tables import provider_symbols


class ProviderSymbolRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def get_provider_symbol(self, source: MarketDataSource, provider_symbol: str) -> ProviderSymbol | None:
        query = select(provider_symbols).where(
            provider_symbols.c.source == source.value,
            provider_symbols.c.provider_symbol == provider_symbol.strip().upper(),
        )
        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            return None
        return _provider_symbol_from_row(row)


def _provider_symbol_from_row(row: Mapping[str, object]) -> ProviderSymbol:
    return ProviderSymbol(
        source=MarketDataSource(str(row["source"])),
        canonical_symbol=str(row["canonical_symbol"]),
        provider_symbol=str(row["provider_symbol"]),
        status=ProviderSymbolStatus(str(row["status"])),
        supported_timeframes=tuple(str(timeframe) for timeframe in row["supported_timeframes"]),
        max_backfill_days=row["max_backfill_days"],
        metadata=row["metadata_json"],
    )
