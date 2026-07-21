from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from market_data_service.domain.enums import MarketDataSource, MarketSymbolStatus, ProviderSymbolStatus
from market_data_service.domain.symbol_normalization import normalize_symbol, split_symbol

SUPPORTED_TIMEFRAMES = ("1h", "4h", "1d")
MAX_BACKFILL_DAYS = 1095


@dataclass(frozen=True, slots=True)
class MarketSymbolSeed:
    canonical_symbol: str
    base_asset: str
    quote_asset: str
    status: MarketSymbolStatus = MarketSymbolStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class ProviderSymbolSeed:
    source: MarketDataSource
    canonical_symbol: str
    provider_symbol: str
    status: ProviderSymbolStatus = ProviderSymbolStatus.TRADING
    supported_timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES
    max_backfill_days: int | None = MAX_BACKFILL_DAYS
    metadata: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class SymbolRegistrySyncResult:
    market_symbols: int
    provider_symbols: int


class SymbolRegistrySyncRepositoryPort(Protocol):
    async def sync_symbols(
        self,
        *,
        market_symbols: tuple[MarketSymbolSeed, ...],
        provider_symbols: tuple[ProviderSymbolSeed, ...],
    ) -> SymbolRegistrySyncResult: ...


DEFAULT_MARKET_SYMBOLS = tuple(
    MarketSymbolSeed(canonical_symbol=f"{base_asset}USDT", base_asset=base_asset, quote_asset="USDT")
    for base_asset in ("BTC", "ETH", "LTC", "SOL", "SUI", "TAO", "XRP")
)

DEFAULT_PROVIDER_SYMBOLS = tuple(
    ProviderSymbolSeed(
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.canonical_symbol,
        metadata={},
    )
    for symbol in DEFAULT_MARKET_SYMBOLS
)


class SymbolRegistrySyncService:
    def __init__(self, repository: SymbolRegistrySyncRepositoryPort) -> None:
        self.repository = repository

    async def sync_default_symbols(self) -> SymbolRegistrySyncResult:
        return await self.repository.sync_symbols(
            market_symbols=DEFAULT_MARKET_SYMBOLS,
            provider_symbols=DEFAULT_PROVIDER_SYMBOLS,
        )

    async def sync_resolved_symbols(
        self,
        *,
        resolved_symbols: tuple[ProviderSymbolSeed, ...],
    ) -> SymbolRegistrySyncResult:
        market_symbol_rows = tuple(
            _market_symbol_seed(provider_symbol.canonical_symbol)
            for provider_symbol in resolved_symbols
        )
        return await self.repository.sync_symbols(
            market_symbols=market_symbol_rows,
            provider_symbols=resolved_symbols,
        )


def build_provider_symbol_seed(
    *,
    source: MarketDataSource,
    canonical_symbol: str,
    provider_symbol: str,
    supported_timeframes: tuple[str, ...],
) -> ProviderSymbolSeed:
    return ProviderSymbolSeed(
        source=source,
        canonical_symbol=normalize_symbol(canonical_symbol),
        provider_symbol=normalize_symbol(provider_symbol),
        supported_timeframes=tuple(timeframe.strip().lower() for timeframe in supported_timeframes),
        metadata={},
    )


def _market_symbol_seed(symbol: str) -> MarketSymbolSeed:
    canonical_symbol = normalize_symbol(symbol)
    base_asset, quote_asset = split_symbol(canonical_symbol)
    return MarketSymbolSeed(
        canonical_symbol=canonical_symbol,
        base_asset=base_asset,
        quote_asset=quote_asset,
    )
