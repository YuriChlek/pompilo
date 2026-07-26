from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from market_data_service.application.services.multi_provider_symbol_resolver import MultiProviderSymbolResolver
from market_data_service.application.services.symbol_registry_sync_service import (
    SymbolRegistrySyncResult,
    SymbolRegistrySyncService,
    build_provider_symbol_seed,
)
from market_data_service.domain.symbol_normalization import normalize_symbol


@dataclass(frozen=True, slots=True)
class SymbolStatus:
    symbol: str
    provider: str
    status: str
    candles: bool
    snapshots: bool
    bot_subscribed: bool


@dataclass(frozen=True, slots=True)
class SymbolsEnableResult:
    registry_sync: SymbolRegistrySyncResult
    bootstrap_ticks: int
    statuses: tuple[SymbolStatus, ...]
    bot_instance_updated: bool


class SymbolOperationsRepositoryPort(Protocol):
    async def list_statuses(self, *, symbols: tuple[str, ...], timeframes: tuple[str, ...]) -> tuple[SymbolStatus, ...]: ...

    async def add_bot_instance_subscriptions(
        self,
        *,
        instance_id: str,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
    ) -> bool: ...


class SymbolOperationsService:
    def __init__(
        self,
        *,
        resolver: MultiProviderSymbolResolver,
        registry_sync: SymbolRegistrySyncService,
        operations_repository: SymbolOperationsRepositoryPort,
    ) -> None:
        self.resolver = resolver
        self.registry_sync = registry_sync
        self.operations_repository = operations_repository

    async def sync_symbols(
        self,
        *,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
    ) -> SymbolRegistrySyncResult:
        normalized_symbols = tuple(normalize_symbol(symbol) for symbol in symbols)
        normalized_timeframes = tuple(timeframe.strip().lower() for timeframe in timeframes)
        provider_rows = []
        for symbol in normalized_symbols:
            resolved = await self.resolver.resolve(symbol)
            if resolved is None:
                continue
            provider_rows.append(
                build_provider_symbol_seed(
                    source=resolved.source,
                    canonical_symbol=symbol,
                    provider_symbol=resolved.provider_symbol,
                    supported_timeframes=normalized_timeframes,
                )
            )
        return await self.registry_sync.sync_resolved_symbols(resolved_symbols=tuple(provider_rows))

    async def list_statuses(
        self,
        *,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
    ) -> tuple[SymbolStatus, ...]:
        return await self.operations_repository.list_statuses(
            symbols=tuple(normalize_symbol(symbol) for symbol in symbols),
            timeframes=tuple(timeframe.strip().lower() for timeframe in timeframes),
        )

    async def enable_symbols(
        self,
        *,
        symbols: tuple[str, ...],
        timeframes: tuple[str, ...],
        bootstrap_ticks: int,
        bot_instance_id: str | None,
        run_bootstrap_tick,
    ) -> SymbolsEnableResult:
        normalized_symbols = tuple(normalize_symbol(symbol) for symbol in symbols)
        normalized_timeframes = tuple(timeframe.strip().lower() for timeframe in timeframes)
        registry_result = await self.sync_symbols(symbols=normalized_symbols, timeframes=normalized_timeframes)

        bot_instance_updated = False
        if bot_instance_id is not None:
            bot_instance_updated = await self.operations_repository.add_bot_instance_subscriptions(
                instance_id=bot_instance_id,
                symbols=normalized_symbols,
                timeframes=normalized_timeframes,
            )

        completed_ticks = 0
        for _ in range(max(0, bootstrap_ticks)):
            await run_bootstrap_tick()
            completed_ticks += 1

        statuses = await self.list_statuses(symbols=normalized_symbols, timeframes=normalized_timeframes)
        return SymbolsEnableResult(
            registry_sync=registry_result,
            bootstrap_ticks=completed_ticks,
            statuses=statuses,
            bot_instance_updated=bot_instance_updated,
        )
