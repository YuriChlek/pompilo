from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from market_data_service.application.market_data_ports import CandleProviderPort, CandleWriterPort
from market_data_service.application.services.multi_provider_symbol_resolver import MultiProviderSymbolResolver, ResolvedProvider
from market_data_service.domain.enums import MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol


@dataclass(frozen=True, slots=True)
class CandlesGetCommand:
    from_time: datetime
    to_time: datetime
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    provider: str


@dataclass(frozen=True, slots=True)
class CandlesGetItemResult:
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    timeframe: str
    fetched_count: int
    inserted_count: int
    skipped_duplicate_count: int


@dataclass(frozen=True, slots=True)
class CandlesGetResult:
    from_time: datetime
    to_time: datetime
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    provider: str
    total_fetched_count: int
    total_inserted_count: int
    total_skipped_duplicate_count: int
    unresolved_symbols: tuple[str, ...]
    items: tuple[CandlesGetItemResult, ...]


class CandlesGetFetchService:
    def __init__(
        self,
        *,
        resolver: MultiProviderSymbolResolver,
        candle_provider: CandleProviderPort,
        candle_writer: CandleWriterPort,
    ) -> None:
        self.resolver = resolver
        self.candle_provider = candle_provider
        self.candle_writer = candle_writer

    async def fetch_candles(
        self,
        *,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        provider: str,
        from_time: datetime,
        to_time: datetime,
    ) -> CandlesGetResult:
        items: list[CandlesGetItemResult] = []
        unresolved_symbols: list[str] = []

        for requested_symbol in symbols:
            if provider == "binance":
                resolved = ResolvedProvider(
                    source=MarketDataSource.BINANCE_SPOT,
                    provider_symbol=requested_symbol.replace("/", "").upper(),
                )
            elif provider == "bybit":
                resolved = ResolvedProvider(
                    source=MarketDataSource.BYBIT_SPOT,
                    provider_symbol=requested_symbol.replace("/", "").upper(),
                )
            else:
                resolved = await self.resolver.resolve(requested_symbol)

            if resolved is None:
                unresolved_symbols.append(requested_symbol)
                continue

            provider_symbol_obj = ProviderSymbol(
                source=resolved.source,
                canonical_symbol=requested_symbol,
                provider_symbol=resolved.provider_symbol,
                status=ProviderSymbolStatus.TRADING,
                supported_timeframes=timeframes,
            )

            for timeframe in timeframes:
                fetched = await self.candle_provider.fetch_closed_candles(
                    provider_symbol_obj,
                    timeframe=timeframe,
                    from_time=from_time,
                    to_time=to_time,
                )
                inserted_count = await self.candle_writer.insert_closed_candles(fetched)
                items.append(
                    CandlesGetItemResult(
                        source=resolved.source,
                        canonical_symbol=requested_symbol,
                        provider_symbol=resolved.provider_symbol,
                        timeframe=timeframe,
                        fetched_count=len(fetched),
                        inserted_count=inserted_count,
                        skipped_duplicate_count=len(fetched) - inserted_count,
                    )
                )

        return CandlesGetResult(
            from_time=from_time,
            to_time=to_time,
            symbols=symbols,
            timeframes=timeframes,
            provider=provider,
            total_fetched_count=sum(item.fetched_count for item in items),
            total_inserted_count=sum(item.inserted_count for item in items),
            total_skipped_duplicate_count=sum(item.skipped_duplicate_count for item in items),
            unresolved_symbols=tuple(unresolved_symbols),
            items=tuple(items),
        )
