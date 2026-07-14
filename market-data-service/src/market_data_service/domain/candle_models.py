from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from market_data_service.domain.enums import MarketDataSource


@dataclass(frozen=True, slots=True)
class CanonicalCandle:
    candle_id: str
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    quote_volume: Decimal | None
    taker_buy_base_volume: Decimal | None
    taker_buy_quote_volume: Decimal | None
    taker_sell_base_volume: Decimal | None
    taker_sell_quote_volume: Decimal | None
    trades_count: int | None
    is_closed: bool
    provider_payload_hash: str
