from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from market_data_service.domain.enums import MarketDataSource, MarketSymbolStatus, ProviderSymbolStatus

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
    MarketSymbolSeed(canonical_symbol=f"{base_asset}/USDT", base_asset=base_asset, quote_asset="USDT")
    for base_asset in ("BTC", "ETH", "LTC", "SOL", "SUI", "TAO", "XRP")
)

DEFAULT_PROVIDER_SYMBOLS = tuple(
    ProviderSymbolSeed(
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol=symbol.canonical_symbol,
        provider_symbol=symbol.canonical_symbol.replace("/", ""),
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
