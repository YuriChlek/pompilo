from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolAvailabilityStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbolAvailability
from market_data_service.domain.availability_rules import AvailabilityCachePolicy


class AvailabilityCachePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = AvailabilityCachePolicy(
            availability_ttl_hours=12.0,
            unsupported_recheck_hours=24.0,
            temporary_error_recheck_minutes=15.0,
        )

    def test_is_cache_valid_returns_true_before_next_check(self) -> None:
        availability = ProviderSymbolAvailability(
            source=MarketDataSource.BINANCE_SPOT,
            requested_symbol="ETH/USDT",
            status=ProviderSymbolAvailabilityStatus.SUPPORTED,
            next_check_at=datetime(2026, 7, 14, 12, tzinfo=UTC),
        )
        now = datetime(2026, 7, 14, 11, 59, 59, tzinfo=UTC)
        self.assertTrue(self.policy.is_cache_valid(availability, now))

    def test_is_cache_valid_returns_false_at_or_after_next_check(self) -> None:
        availability = ProviderSymbolAvailability(
            source=MarketDataSource.BINANCE_SPOT,
            requested_symbol="ETH/USDT",
            status=ProviderSymbolAvailabilityStatus.SUPPORTED,
            next_check_at=datetime(2026, 7, 14, 12, tzinfo=UTC),
        )
        now = datetime(2026, 7, 14, 12, tzinfo=UTC)
        self.assertFalse(self.policy.is_cache_valid(availability, now))

        now = datetime(2026, 7, 14, 13, tzinfo=UTC)
        self.assertFalse(self.policy.is_cache_valid(availability, now))

    def test_calculate_next_check_supported(self) -> None:
        checked_at = datetime(2026, 7, 14, 10, tzinfo=UTC)
        next_check = self.policy.calculate_next_check(ProviderSymbolAvailabilityStatus.SUPPORTED, checked_at)
        self.assertEqual(next_check, checked_at + timedelta(hours=12))

    def test_calculate_next_check_unsupported(self) -> None:
        checked_at = datetime(2026, 7, 14, 10, tzinfo=UTC)
        next_check = self.policy.calculate_next_check(ProviderSymbolAvailabilityStatus.UNSUPPORTED, checked_at)
        self.assertEqual(next_check, checked_at + timedelta(hours=24))

    def test_calculate_next_check_temporary_error(self) -> None:
        checked_at = datetime(2026, 7, 14, 10, tzinfo=UTC)
        next_check = self.policy.calculate_next_check(ProviderSymbolAvailabilityStatus.TEMPORARY_ERROR, checked_at)
        self.assertEqual(next_check, checked_at + timedelta(minutes=15))

    def test_create_availability(self) -> None:
        now = datetime(2026, 7, 14, 10, tzinfo=UTC)
        availability = self.policy.create_availability(
            source=MarketDataSource.BINANCE_SPOT,
            requested_symbol="ETH/USDT",
            status=ProviderSymbolAvailabilityStatus.UNSUPPORTED,
            now=now,
            failure_reason="Not listed",
            metadata={"source": "api"},
        )

        self.assertEqual(availability.source, MarketDataSource.BINANCE_SPOT)
        self.assertEqual(availability.requested_symbol, "ETH/USDT")
        self.assertEqual(availability.status, ProviderSymbolAvailabilityStatus.UNSUPPORTED)
        self.assertEqual(availability.first_seen_at, now)
        self.assertEqual(availability.last_checked_at, now)
        self.assertEqual(availability.next_check_at, now + timedelta(hours=24))
        self.assertEqual(availability.failure_reason, "Not listed")
        self.assertEqual(availability.metadata, {"source": "api"})
