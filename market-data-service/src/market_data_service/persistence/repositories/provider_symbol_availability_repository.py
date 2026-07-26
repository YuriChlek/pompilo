from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolAvailabilityStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbolAvailability
from market_data_service.persistence.tables.provider_symbol_availability_tables import provider_symbol_availability


class ProviderSymbolAvailabilityRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def get(
        self,
        source: MarketDataSource,
        requested_symbol: str,
    ) -> ProviderSymbolAvailability | None:
        query = select(provider_symbol_availability).where(
            provider_symbol_availability.c.source == source.value,
            provider_symbol_availability.c.requested_symbol == requested_symbol.strip().upper(),
        )
        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            return None
        return _from_row(row)

    async def upsert(self, availability: ProviderSymbolAvailability) -> None:
        stmt = insert(provider_symbol_availability).values(
            source=availability.source.value,
            requested_symbol=availability.requested_symbol,
            provider_symbol=availability.provider_symbol,
            status=availability.status.value,
            next_check_at=availability.next_check_at,
            failure_reason=availability.failure_reason,
            metadata_json=availability.metadata or {},
        )
        await self.connection.execute(
            stmt.on_conflict_do_update(
                index_elements=["source", "requested_symbol"],
                set_={
                    "provider_symbol": stmt.excluded.provider_symbol,
                    "status": stmt.excluded.status,
                    "last_checked_at": func.now(),
                    "next_check_at": stmt.excluded.next_check_at,
                    "failure_reason": stmt.excluded.failure_reason,
                    "metadata_json": stmt.excluded.metadata_json,
                },
            )
        )


def _from_row(row: Mapping[str, object]) -> ProviderSymbolAvailability:
    return ProviderSymbolAvailability(
        source=MarketDataSource(str(row["source"])),
        requested_symbol=str(row["requested_symbol"]),
        provider_symbol=str(row["provider_symbol"]) if row["provider_symbol"] is not None else None,
        status=ProviderSymbolAvailabilityStatus(str(row["status"])),
        first_seen_at=row["first_seen_at"],
        last_checked_at=row["last_checked_at"],
        next_check_at=row["next_check_at"],
        failure_reason=str(row["failure_reason"]) if row["failure_reason"] is not None else None,
        metadata=row["metadata_json"],
    )
