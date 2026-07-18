from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolAvailabilityStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbol, ProviderSymbolAvailability
from market_data_service.domain.availability_rules import AvailabilityCachePolicy
from market_data_service.application.services.multi_provider_symbol_resolver import (
    MultiProviderSymbolResolver,
    ResolvedProvider,
)
from market_data_service.infrastructure.providers.provider_errors import NonRetryableProviderError, RetryableProviderError


class MockAvailabilityRepository:
    def __init__(self) -> None:
        self.records: dict[tuple[MarketDataSource, str], ProviderSymbolAvailability] = {}
        self.get_calls: list[tuple[MarketDataSource, str]] = []
        self.upsert_calls: list[ProviderSymbolAvailability] = []

    async def get(self, source: MarketDataSource, requested_symbol: str) -> ProviderSymbolAvailability | None:
        self.get_calls.append((source, requested_symbol))
        return self.records.get((source, requested_symbol))

    async def upsert(self, availability: ProviderSymbolAvailability) -> None:
        self.upsert_calls.append(availability)
        self.records[(availability.source, availability.requested_symbol)] = availability


class MockCandleProvider:
    def __init__(self, responses: dict[str, object]) -> None:
        self.responses = responses
        self.calls: list[tuple[ProviderSymbol, str, datetime, datetime]] = []

    async def fetch_closed_candles(
        self,
        provider_symbol: ProviderSymbol,
        *,
        timeframe: str,
        from_time: datetime,
        to_time: datetime,
    ) -> list[object]:
        self.calls.append((provider_symbol, timeframe, from_time, to_time))
        response = self.responses.get(provider_symbol.provider_symbol)
        if isinstance(response, Exception):
            raise response
        return response or []


class MultiProviderSymbolResolverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 14, 10, tzinfo=UTC)
        self.policy = AvailabilityCachePolicy(
            availability_ttl_hours=24,
            unsupported_recheck_hours=24,
            temporary_error_recheck_minutes=5,
        )
        self.repository = MockAvailabilityRepository()

    async def test_resolve_cache_hit_supported(self) -> None:
        # Pre-populate cache with SUPPORTED
        availability = ProviderSymbolAvailability(
            source=MarketDataSource.BINANCE_SPOT,
            requested_symbol="BTC/USDT",
            provider_symbol="BTCUSDT",
            status=ProviderSymbolAvailabilityStatus.SUPPORTED,
            next_check_at=self.now + timedelta(hours=1),
        )
        self.repository.records[(MarketDataSource.BINANCE_SPOT, "BTC/USDT")] = availability

        binance_adapter = MockCandleProvider({})
        resolver = MultiProviderSymbolResolver(
            availability_repository=self.repository,
            adapters={MarketDataSource.BINANCE_SPOT: binance_adapter},
            priority=("binance",),
            cache_policy=self.policy,
            now_provider=lambda: self.now,
        )

        resolved = await resolver.resolve("BTC/USDT")

        self.assertEqual(resolved, ResolvedProvider(MarketDataSource.BINANCE_SPOT, "BTCUSDT"))
        self.assertEqual(len(binance_adapter.calls), 0)
        self.assertEqual(len(self.repository.get_calls), 1)

    async def test_resolve_cache_hit_unsupported_skips_provider(self) -> None:
        # Pre-populate Binance cache with UNSUPPORTED
        self.repository.records[(MarketDataSource.BINANCE_SPOT, "HYPE/USDT")] = ProviderSymbolAvailability(
            source=MarketDataSource.BINANCE_SPOT,
            requested_symbol="HYPE/USDT",
            status=ProviderSymbolAvailabilityStatus.UNSUPPORTED,
            next_check_at=self.now + timedelta(hours=1),
        )

        binance_adapter = MockCandleProvider({})
        bybit_adapter = MockCandleProvider({"HYPEUSDT": []})
        resolver = MultiProviderSymbolResolver(
            availability_repository=self.repository,
            adapters={
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            },
            priority=("binance", "bybit"),
            cache_policy=self.policy,
            now_provider=lambda: self.now,
        )

        resolved = await resolver.resolve("HYPE/USDT")

        # Resolves to Bybit because Binance is skipped due to cached UNSUPPORTED
        self.assertEqual(resolved, ResolvedProvider(MarketDataSource.BYBIT_SPOT, "HYPEUSDT"))
        self.assertEqual(len(binance_adapter.calls), 0)
        self.assertEqual(len(bybit_adapter.calls), 1)

    async def test_resolve_cache_miss_queries_provider_success(self) -> None:
        binance_adapter = MockCandleProvider({"BTCUSDT": []})
        resolver = MultiProviderSymbolResolver(
            availability_repository=self.repository,
            adapters={MarketDataSource.BINANCE_SPOT: binance_adapter},
            priority=("binance",),
            cache_policy=self.policy,
            now_provider=lambda: self.now,
        )

        resolved = await resolver.resolve("BTC/USDT")

        self.assertEqual(resolved, ResolvedProvider(MarketDataSource.BINANCE_SPOT, "BTCUSDT"))
        self.assertEqual(len(binance_adapter.calls), 1)
        self.assertEqual(len(self.repository.upsert_calls), 1)
        self.assertEqual(self.repository.upsert_calls[0].status, ProviderSymbolAvailabilityStatus.SUPPORTED)

    async def test_resolve_cache_miss_queries_provider_unsupported(self) -> None:
        # Binance adapter raises NonRetryableProviderError (symbol unsupported)
        binance_adapter = MockCandleProvider({"BTCUSDT": NonRetryableProviderError("Invalid symbol")})
        bybit_adapter = MockCandleProvider({"BTCUSDT": []})
        resolver = MultiProviderSymbolResolver(
            availability_repository=self.repository,
            adapters={
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            },
            priority=("binance", "bybit"),
            cache_policy=self.policy,
            now_provider=lambda: self.now,
        )

        resolved = await resolver.resolve("BTC/USDT")

        # Should fall back to Bybit
        self.assertEqual(resolved, ResolvedProvider(MarketDataSource.BYBIT_SPOT, "BTCUSDT"))
        self.assertEqual(len(binance_adapter.calls), 1)
        self.assertEqual(len(bybit_adapter.calls), 1)

        # Verify Binance cache was updated as UNSUPPORTED
        self.assertEqual(self.repository.upsert_calls[0].source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(self.repository.upsert_calls[0].status, ProviderSymbolAvailabilityStatus.UNSUPPORTED)

    async def test_resolve_cache_miss_queries_provider_temporary_error(self) -> None:
        # Binance adapter raises RetryableProviderError (temporary error)
        binance_adapter = MockCandleProvider({"BTCUSDT": RetryableProviderError("Rate limit")})
        bybit_adapter = MockCandleProvider({"BTCUSDT": []})
        resolver = MultiProviderSymbolResolver(
            availability_repository=self.repository,
            adapters={
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            },
            priority=("binance", "bybit"),
            cache_policy=self.policy,
            now_provider=lambda: self.now,
        )

        resolved = await resolver.resolve("BTC/USDT")

        # Should fall back to Bybit
        self.assertEqual(resolved, ResolvedProvider(MarketDataSource.BYBIT_SPOT, "BTCUSDT"))
        self.assertEqual(len(binance_adapter.calls), 1)
        self.assertEqual(len(bybit_adapter.calls), 1)

        # Verify Binance cache was updated as TEMPORARY_ERROR
        self.assertEqual(self.repository.upsert_calls[0].source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(self.repository.upsert_calls[0].status, ProviderSymbolAvailabilityStatus.TEMPORARY_ERROR)

    async def test_resolve_all_fail_returns_none(self) -> None:
        binance_adapter = MockCandleProvider({"BTCUSDT": NonRetryableProviderError("Invalid symbol")})
        bybit_adapter = MockCandleProvider({"BTCUSDT": NonRetryableProviderError("Invalid symbol")})
        resolver = MultiProviderSymbolResolver(
            availability_repository=self.repository,
            adapters={
                MarketDataSource.BINANCE_SPOT: binance_adapter,
                MarketDataSource.BYBIT_SPOT: bybit_adapter,
            },
            priority=("binance", "bybit"),
            cache_policy=self.policy,
            now_provider=lambda: self.now,
        )

        resolved = await resolver.resolve("BTC/USDT")

        self.assertIsNone(resolved)
        self.assertEqual(len(binance_adapter.calls), 1)
        self.assertEqual(len(bybit_adapter.calls), 1)
