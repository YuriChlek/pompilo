from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.collection_models import CandleCollectionState
from market_data_service.domain.enums import MarketDataSource
from market_data_service.persistence.tables.collection_states_tables import collection_states


class CollectionStateRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def get(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
    ) -> CandleCollectionState | None:
        query = select(collection_states).where(
            collection_states.c.source == source.value,
            collection_states.c.canonical_symbol == canonical_symbol,
            collection_states.c.timeframe == timeframe,
        )
        row = (await self.connection.execute(query)).mappings().one_or_none()
        if row is None:
            return None
        return _row_to_state(row)

    async def create_bootstrap_state(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        bootstrap_from: datetime,
        bootstrap_to: datetime,
    ) -> CandleCollectionState:
        statement = (
            insert(collection_states)
            .values(
                source=source.value,
                canonical_symbol=canonical_symbol,
                provider_symbol=provider_symbol,
                timeframe=timeframe,
                bootstrap_from=bootstrap_from,
                bootstrap_to=bootstrap_to,
                bootstrap_next_from=bootstrap_from,
                bootstrap_completed_at=None,
                last_successful_close_time=None,
            )
            .on_conflict_do_nothing(index_elements=["source", "canonical_symbol", "timeframe"])
        )
        await self.connection.execute(statement)
        state = await self.get(source=source, canonical_symbol=canonical_symbol, timeframe=timeframe)
        if state is None:
            raise RuntimeError("Collection state was not created")
        return state

    async def mark_bootstrap_progress(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        provider_symbol: str,
        bootstrap_next_from: datetime,
        completed_at: datetime | None,
        last_successful_close_time: datetime | None,
    ) -> None:
        values = {
            "provider_symbol": provider_symbol,
            "bootstrap_next_from": bootstrap_next_from,
            "bootstrap_completed_at": completed_at,
            "updated_at": completed_at or bootstrap_next_from,
        }
        if last_successful_close_time is not None:
            values["last_successful_close_time"] = last_successful_close_time
        await self.connection.execute(
            collection_states.update()
            .where(
                collection_states.c.source == source.value,
                collection_states.c.canonical_symbol == canonical_symbol,
                collection_states.c.timeframe == timeframe,
            )
            .values(**values)
        )

    async def mark_incremental_progress(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        timeframe: str,
        provider_symbol: str,
        last_successful_close_time: datetime,
        updated_at: datetime,
    ) -> None:
        await self.connection.execute(
            collection_states.update()
            .where(
                collection_states.c.source == source.value,
                collection_states.c.canonical_symbol == canonical_symbol,
                collection_states.c.timeframe == timeframe,
            )
            .values(
                provider_symbol=provider_symbol,
                last_successful_close_time=last_successful_close_time,
                updated_at=updated_at,
            )
        )


def _row_to_state(row: Mapping[str, object]) -> CandleCollectionState:
    return CandleCollectionState(
        source=MarketDataSource(str(row["source"])),
        canonical_symbol=str(row["canonical_symbol"]),
        provider_symbol=str(row["provider_symbol"]),
        timeframe=str(row["timeframe"]),
        bootstrap_from=row["bootstrap_from"],
        bootstrap_to=row["bootstrap_to"],
        bootstrap_next_from=row["bootstrap_next_from"],
        bootstrap_completed_at=row["bootstrap_completed_at"],
        last_successful_close_time=row["last_successful_close_time"],
    )
