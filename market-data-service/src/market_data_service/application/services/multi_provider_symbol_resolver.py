from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Mapping

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolAvailabilityStatus, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol
from market_data_service.domain.availability_rules import AvailabilityCachePolicy
from market_data_service.persistence.repositories.provider_symbol_availability_repository import (
    ProviderSymbolAvailabilityRepository,
)
from market_data_service.application.market_data_ports import CandleProviderPort
from market_data_service.infrastructure.providers.provider_errors import NonRetryableProviderError


class ResolutionFailedError(ValueError):
    """Raised when a symbol cannot be resolved to any provider."""


@dataclass(frozen=True, slots=True)
class ResolvedProvider:
    source: MarketDataSource
    provider_symbol: str


class MultiProviderSymbolResolver:
    def __init__(
        self,
        *,
        availability_repository: ProviderSymbolAvailabilityRepository,
        adapters: Mapping[MarketDataSource, CandleProviderPort],
        priority: tuple[str, ...],
        cache_policy: AvailabilityCachePolicy,
        now_provider=None,
    ) -> None:
        self.availability_repository = availability_repository
        self.adapters = adapters
        self.priority = priority
        self.cache_policy = cache_policy
        self.now_provider = now_provider or (lambda: datetime.now(UTC))

    async def resolve(self, requested_symbol: str) -> ResolvedProvider | None:
        now = self.now_provider()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)

        for provider_name in self.priority:
            source = _map_provider_name_to_source(provider_name)
            if source is None:
                continue

            adapter = self.adapters.get(source)
            if adapter is None:
                continue

            # 1. Check cache
            cached = await self.availability_repository.get(source, requested_symbol)
            if cached is not None and self.cache_policy.is_cache_valid(cached, now):
                if cached.status == ProviderSymbolAvailabilityStatus.SUPPORTED:
                    assert cached.provider_symbol is not None
                    return ResolvedProvider(source=source, provider_symbol=cached.provider_symbol)
                # If UNSUPPORTED or TEMPORARY_ERROR with valid cache, skip this provider
                continue

            # 2. Cache is missing or expired -> Query provider
            provider_symbol_str = requested_symbol.replace("/", "").upper()
            provider_symbol_obj = ProviderSymbol(
                source=source,
                canonical_symbol=requested_symbol,
                provider_symbol=provider_symbol_str,
                status=ProviderSymbolStatus.TRADING,
                supported_timeframes=("1h", "4h", "1d"),
            )

            try:
                # Perform a lightweight test fetch call (last 2 hours)
                await adapter.fetch_closed_candles(
                    provider_symbol_obj,
                    timeframe="1h",
                    from_time=now - timedelta(hours=2),
                    to_time=now,
                )
                status = ProviderSymbolAvailabilityStatus.SUPPORTED
                failure_reason = None
            except NonRetryableProviderError as exc:
                status = ProviderSymbolAvailabilityStatus.UNSUPPORTED
                failure_reason = str(exc)
            except Exception as exc:
                status = ProviderSymbolAvailabilityStatus.TEMPORARY_ERROR
                failure_reason = str(exc)

            # Update cache
            availability = self.cache_policy.create_availability(
                source=source,
                requested_symbol=requested_symbol,
                status=status,
                now=now,
                provider_symbol=provider_symbol_str if status == ProviderSymbolAvailabilityStatus.SUPPORTED else None,
                failure_reason=failure_reason,
            )
            await self.availability_repository.upsert(availability)

            if status == ProviderSymbolAvailabilityStatus.SUPPORTED:
                return ResolvedProvider(source=source, provider_symbol=provider_symbol_str)

        return None


def _map_provider_name_to_source(name: str) -> MarketDataSource | None:
    n = name.strip().lower()
    if n == "binance":
        return MarketDataSource.BINANCE_SPOT
    elif n == "bybit":
        return MarketDataSource.BYBIT_SPOT
    return None
