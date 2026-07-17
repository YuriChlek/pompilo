from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.enums import MarketDataSource
from market_data_service.persistence.tables import market_candles


class GapScanRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def list_closed_candle_open_times(
        self,
        *,
        source: MarketDataSource,
        canonical_symbol: str,
        provider_symbol: str,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> tuple[datetime, ...]:
        query = (
            select(market_candles.c.open_time)
            .where(
                market_candles.c.source == source.value,
                market_candles.c.canonical_symbol == canonical_symbol,
                market_candles.c.provider_symbol == provider_symbol,
                market_candles.c.timeframe == timeframe,
                market_candles.c.is_closed.is_(True),
                market_candles.c.open_time >= from_time,
                market_candles.c.open_time < to_time,
            )
            .order_by(market_candles.c.open_time)
        )
        result = await self.connection.execute(query)
        return tuple(row[0] for row in result)
