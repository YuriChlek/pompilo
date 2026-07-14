from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.persistence.tables import market_candles


class CandleRepository:
    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def insert_closed_candles(self, candles: Sequence[CanonicalCandle]) -> int:
        if not candles:
            return 0

        statement = (
            insert(market_candles)
            .values([_candle_to_row(candle) for candle in candles])
            .on_conflict_do_nothing(
                index_elements=["source", "canonical_symbol", "timeframe", "open_time"],
            )
        )
        result = await self.connection.execute(statement)
        return int(result.rowcount or 0)


def _candle_to_row(candle: CanonicalCandle) -> dict[str, object]:
    return {
        "candle_id": candle.candle_id,
        "source": candle.source.value,
        "canonical_symbol": candle.canonical_symbol,
        "provider_symbol": candle.provider_symbol,
        "timeframe": candle.timeframe,
        "open_time": candle.open_time,
        "close_time": candle.close_time,
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "quote_volume": candle.quote_volume,
        "taker_buy_base_volume": candle.taker_buy_base_volume,
        "taker_buy_quote_volume": candle.taker_buy_quote_volume,
        "taker_sell_base_volume": candle.taker_sell_base_volume,
        "taker_sell_quote_volume": candle.taker_sell_quote_volume,
        "trades_count": candle.trades_count,
        "is_closed": candle.is_closed,
        "provider_payload_hash": candle.provider_payload_hash,
    }
