from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from market_data_service.domain.candle_models import CanonicalCandle
from market_data_service.domain.candle_normalizer import normalize_closed_candle
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.domain.timeframe_rules import get_timeframe_duration


class FixtureCandleProvider:
    """Deterministic candle provider used only by explicit smoke/runtime checks."""

    async def fetch_closed_candles(
        self,
        provider_symbol: ProviderSymbol,
        *,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[CanonicalCandle]:
        if provider_symbol.source != MarketDataSource.BINANCE_SPOT:
            raise ValueError(f"Unsupported fixture source: {provider_symbol.source}")
        normalized_timeframe = timeframe.strip().lower()
        duration = get_timeframe_duration(normalized_timeframe)
        candles: list[CanonicalCandle] = []
        open_time = from_time.astimezone(UTC)
        index = 0
        while open_time + duration <= to_time.astimezone(UTC):
            base_price = Decimal("2500") + Decimal(index)
            candles.append(
                normalize_closed_candle(
                    {
                        "open_time": open_time,
                        "open": base_price,
                        "high": base_price + Decimal("12.50"),
                        "low": base_price - Decimal("8.25"),
                        "close": base_price + Decimal("3.75"),
                        "volume": Decimal("100") + Decimal(index),
                        "quote_volume": Decimal("250000") + Decimal(index),
                        "trades_count": 1000 + index,
                        "taker_buy_base_volume": Decimal("45") + Decimal(index),
                        "taker_buy_quote_volume": Decimal("112500") + Decimal(index),
                    },
                    source=provider_symbol.source,
                    canonical_symbol=provider_symbol.canonical_symbol,
                    provider_symbol=provider_symbol.provider_symbol,
                    timeframe=normalized_timeframe,
                )
            )
            open_time += duration
            index += 1
        return candles
