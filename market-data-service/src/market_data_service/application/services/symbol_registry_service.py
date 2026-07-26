from __future__ import annotations

from collections.abc import Iterable, Mapping

from market_data_service.application.symbol_registry_ports import ProviderSymbolRegistryPort
from market_data_service.domain.enums import MarketDataSource
from market_data_service.domain.symbol_registry_models import ProviderSymbol


class SymbolRegistryCoverageError(ValueError):
    """Raised when configured strategy symbols are not covered by provider mappings."""


class ProviderSymbolNotActiveError(ValueError):
    """Raised when sync is requested for a missing or non-trading provider symbol."""


def validate_provider_registry_coverage(
    configured_provider_symbols: Iterable[str],
    provider_symbols: Iterable[ProviderSymbol],
    *,
    source: MarketDataSource,
    required_timeframes: Iterable[str],
) -> None:
    required_timeframe_set = tuple(required_timeframes)
    mappings: Mapping[str, ProviderSymbol] = {
        symbol.provider_symbol.upper(): symbol
        for symbol in provider_symbols
        if symbol.source == source
    }

    missing: list[str] = []
    inactive: list[str] = []
    unsupported: list[str] = []

    for raw_symbol in configured_provider_symbols:
        provider_symbol = raw_symbol.strip().upper()
        mapping = mappings.get(provider_symbol)
        if mapping is None:
            missing.append(provider_symbol)
            continue
        if not mapping.is_trading:
            inactive.append(provider_symbol)
            continue
        if not mapping.supports_all_timeframes(required_timeframe_set):
            unsupported.append(provider_symbol)

    errors: list[str] = []
    if missing:
        errors.append(f"missing provider mappings: {','.join(sorted(missing))}")
    if inactive:
        errors.append(f"inactive provider mappings: {','.join(sorted(inactive))}")
    if unsupported:
        errors.append(f"missing required timeframes: {','.join(sorted(unsupported))}")
    if errors:
        raise SymbolRegistryCoverageError("; ".join(errors))


async def assert_sync_mapping_active(
    registry: ProviderSymbolRegistryPort,
    *,
    source: MarketDataSource,
    provider_symbol: str,
    required_timeframe: str,
) -> ProviderSymbol:
    normalized_provider_symbol = provider_symbol.strip().upper()
    mapping = await registry.get_provider_symbol(source, normalized_provider_symbol)
    if mapping is None:
        raise ProviderSymbolNotActiveError(f"No provider mapping for {source}:{normalized_provider_symbol}")
    if not mapping.is_trading:
        raise ProviderSymbolNotActiveError(f"Provider mapping is not trading: {source}:{normalized_provider_symbol}")
    if not mapping.supports_all_timeframes((required_timeframe,)):
        raise ProviderSymbolNotActiveError(
            f"Provider mapping does not support {required_timeframe}: {source}:{normalized_provider_symbol}"
        )
    return mapping
