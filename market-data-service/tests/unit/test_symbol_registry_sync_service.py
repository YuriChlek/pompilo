from __future__ import annotations

import unittest

from market_data_service.application.services.symbol_registry_sync_service import (
    DEFAULT_MARKET_SYMBOLS,
    DEFAULT_PROVIDER_SYMBOLS,
    SymbolRegistrySyncResult,
    SymbolRegistrySyncService,
)


class SymbolRegistrySyncServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_default_symbols_is_idempotent_for_repeated_runs(self) -> None:
        repository = FakeSymbolRegistryRepository()
        service = SymbolRegistrySyncService(repository)

        first_result = await service.sync_default_symbols()
        second_result = await service.sync_default_symbols()

        self.assertEqual(first_result, SymbolRegistrySyncResult(market_symbols=7, provider_symbols=7))
        self.assertEqual(second_result, SymbolRegistrySyncResult(market_symbols=7, provider_symbols=7))
        self.assertEqual(len(repository.market_symbols_by_canonical_symbol), 7)
        self.assertEqual(len(repository.provider_symbols_by_source_and_canonical_symbol), 7)
        self.assertEqual(
            set(repository.market_symbols_by_canonical_symbol),
            {"BTC/USDT", "ETH/USDT", "LTC/USDT", "SOL/USDT", "SUI/USDT", "TAO/USDT", "XRP/USDT"},
        )
        self.assertEqual(
            {
                symbol.provider_symbol
                for symbol in repository.provider_symbols_by_source_and_canonical_symbol.values()
            },
            {"BTCUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "SUIUSDT", "TAOUSDT", "XRPUSDT"},
        )

    def test_default_symbol_seeds_have_unique_natural_keys(self) -> None:
        self.assertEqual(
            len({symbol.canonical_symbol for symbol in DEFAULT_MARKET_SYMBOLS}),
            len(DEFAULT_MARKET_SYMBOLS),
        )
        self.assertEqual(
            len({(symbol.source, symbol.canonical_symbol) for symbol in DEFAULT_PROVIDER_SYMBOLS}),
            len(DEFAULT_PROVIDER_SYMBOLS),
        )
        self.assertEqual(
            len({(symbol.source, symbol.provider_symbol) for symbol in DEFAULT_PROVIDER_SYMBOLS}),
            len(DEFAULT_PROVIDER_SYMBOLS),
        )


class FakeSymbolRegistryRepository:
    def __init__(self) -> None:
        self.market_symbols_by_canonical_symbol = {}
        self.provider_symbols_by_source_and_canonical_symbol = {}

    async def sync_symbols(self, *, market_symbols, provider_symbols):
        for symbol in market_symbols:
            self.market_symbols_by_canonical_symbol[symbol.canonical_symbol] = symbol
        for symbol in provider_symbols:
            self.provider_symbols_by_source_and_canonical_symbol[(symbol.source, symbol.canonical_symbol)] = symbol
        return SymbolRegistrySyncResult(
            market_symbols=len(self.market_symbols_by_canonical_symbol),
            provider_symbols=len(self.provider_symbols_by_source_and_canonical_symbol),
        )
