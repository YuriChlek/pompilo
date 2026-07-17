from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.application.services.symbol_registry_sync_service import (
    MarketSymbolSeed,
    ProviderSymbolSeed,
    SymbolRegistrySyncResult,
)
from market_data_service.persistence.tables import market_symbols, provider_symbols


class SymbolRegistryRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def sync_symbols(
        self,
        *,
        market_symbols: tuple[MarketSymbolSeed, ...],
        provider_symbols: tuple[ProviderSymbolSeed, ...],
    ) -> SymbolRegistrySyncResult:
        async with self.connection.begin():
            market_count = await self._upsert_market_symbols(market_symbols)
            provider_count = await self._upsert_provider_symbols(provider_symbols)
        return SymbolRegistrySyncResult(market_symbols=market_count, provider_symbols=provider_count)

    async def _upsert_market_symbols(self, rows: tuple[MarketSymbolSeed, ...]) -> int:
        if not rows:
            return 0

        values = [
            {
                "canonical_symbol": row.canonical_symbol,
                "base_asset": row.base_asset,
                "quote_asset": row.quote_asset,
                "status": row.status.value,
            }
            for row in rows
        ]
        statement = insert(market_symbols).values(values)
        result = await self.connection.execute(
            statement.on_conflict_do_update(
                index_elements=["canonical_symbol"],
                set_={
                    "base_asset": statement.excluded.base_asset,
                    "quote_asset": statement.excluded.quote_asset,
                    "status": statement.excluded.status,
                    "updated_at": func.now(),
                },
            )
        )
        return result.rowcount or 0

    async def _upsert_provider_symbols(self, rows: tuple[ProviderSymbolSeed, ...]) -> int:
        if not rows:
            return 0

        values = [
            {
                "source": row.source.value,
                "canonical_symbol": row.canonical_symbol,
                "provider_symbol": row.provider_symbol,
                "status": row.status.value,
                "supported_timeframes": list(row.supported_timeframes),
                "min_available_time": None,
                "max_backfill_days": row.max_backfill_days,
                "metadata_json": row.metadata or {},
            }
            for row in rows
        ]
        statement = insert(provider_symbols).values(values)
        result = await self.connection.execute(
            statement.on_conflict_do_update(
                index_elements=["source", "canonical_symbol"],
                set_={
                    "provider_symbol": statement.excluded.provider_symbol,
                    "status": statement.excluded.status,
                    "supported_timeframes": statement.excluded.supported_timeframes,
                    "min_available_time": statement.excluded.min_available_time,
                    "max_backfill_days": statement.excluded.max_backfill_days,
                    "metadata_json": statement.excluded.metadata_json,
                    "updated_at": func.now(),
                },
            )
        )
        return result.rowcount or 0
