from __future__ import annotations

import unittest

from market_data_service.application.services.symbol_registry_service import (
    ProviderSymbolNotActiveError,
    SymbolRegistryCoverageError,
    assert_sync_mapping_active,
    validate_provider_registry_coverage,
)
from market_data_service.domain.enums import MarketDataSource, ProviderSymbolStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol

SUPPORTED_TIMEFRAMES = ("1h", "4h", "1d")
CURRENT_TRADING_PROVIDER_SYMBOLS = ("BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "XRPUSDT")
BINANCE_SPOT_PROVIDER_SYMBOLS = tuple(
    ProviderSymbol(
        source=MarketDataSource.BINANCE_SPOT,
        canonical_symbol=provider_symbol.replace("USDT", "/USDT"),
        provider_symbol=provider_symbol,
        status=ProviderSymbolStatus.TRADING,
        supported_timeframes=SUPPORTED_TIMEFRAMES,
    )
    for provider_symbol in CURRENT_TRADING_PROVIDER_SYMBOLS
)


class FakeProviderSymbolRegistry:
    def __init__(self, mapping: ProviderSymbol | None) -> None:
        self.mapping = mapping

    async def get_provider_symbol(self, source: MarketDataSource, provider_symbol: str) -> ProviderSymbol | None:
        return self.mapping


class SymbolRegistryServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_default_registry_covers_current_bot_symbols_and_timeframes(self) -> None:
        validate_provider_registry_coverage(
            CURRENT_TRADING_PROVIDER_SYMBOLS,
            BINANCE_SPOT_PROVIDER_SYMBOLS,
            source=MarketDataSource.BINANCE_SPOT,
            required_timeframes=SUPPORTED_TIMEFRAMES,
        )

    def test_registry_coverage_blocks_missing_provider_mapping(self) -> None:
        with self.assertRaisesRegex(SymbolRegistryCoverageError, "missing provider mappings: DOGEUSDT"):
            validate_provider_registry_coverage(
                (*CURRENT_TRADING_PROVIDER_SYMBOLS, "DOGEUSDT"),
                BINANCE_SPOT_PROVIDER_SYMBOLS,
                source=MarketDataSource.BINANCE_SPOT,
                required_timeframes=SUPPORTED_TIMEFRAMES,
            )

    def test_registry_coverage_blocks_inactive_provider_mapping(self) -> None:
        inactive_mapping = ProviderSymbol(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            status=ProviderSymbolStatus.HALTED,
            supported_timeframes=SUPPORTED_TIMEFRAMES,
        )

        with self.assertRaisesRegex(SymbolRegistryCoverageError, "inactive provider mappings: ETHUSDT"):
            validate_provider_registry_coverage(
                ("ETHUSDT",),
                (inactive_mapping,),
                source=MarketDataSource.BINANCE_SPOT,
                required_timeframes=SUPPORTED_TIMEFRAMES,
            )

    def test_registry_coverage_blocks_missing_required_timeframe(self) -> None:
        partial_mapping = ProviderSymbol(
            source=MarketDataSource.BINANCE_SPOT,
            canonical_symbol="ETH/USDT",
            provider_symbol="ETHUSDT",
            status=ProviderSymbolStatus.TRADING,
            supported_timeframes=("1h", "4h"),
        )

        with self.assertRaisesRegex(SymbolRegistryCoverageError, "missing required timeframes: ETHUSDT"):
            validate_provider_registry_coverage(
                ("ETHUSDT",),
                (partial_mapping,),
                source=MarketDataSource.BINANCE_SPOT,
                required_timeframes=SUPPORTED_TIMEFRAMES,
            )

    async def test_sync_mapping_guard_returns_active_mapping(self) -> None:
        registry = FakeProviderSymbolRegistry(BINANCE_SPOT_PROVIDER_SYMBOLS[0])

        mapping = await assert_sync_mapping_active(
            registry,
            source=MarketDataSource.BINANCE_SPOT,
            provider_symbol=BINANCE_SPOT_PROVIDER_SYMBOLS[0].provider_symbol,
            required_timeframe="1h",
        )

        self.assertEqual(mapping.status, ProviderSymbolStatus.TRADING)

    async def test_sync_mapping_guard_blocks_missing_mapping(self) -> None:
        registry = FakeProviderSymbolRegistry(None)

        with self.assertRaises(ProviderSymbolNotActiveError):
            await assert_sync_mapping_active(
                registry,
                source=MarketDataSource.BINANCE_SPOT,
                provider_symbol="DOGEUSDT",
                required_timeframe="1h",
            )
