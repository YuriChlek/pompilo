from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping

from market_data_service.domain.enums import MarketDataSource, ProviderSymbolAvailabilityStatus
from market_data_service.domain.symbol_registry_models import ProviderSymbolAvailability


class AvailabilityCachePolicy:
    def __init__(
        self,
        *,
        availability_ttl_hours: float = 24.0,
        unsupported_recheck_hours: float = 24.0,
        temporary_error_recheck_minutes: float = 5.0,
    ) -> None:
        self.availability_ttl_hours = availability_ttl_hours
        self.unsupported_recheck_hours = unsupported_recheck_hours
        self.temporary_error_recheck_minutes = temporary_error_recheck_minutes

    def is_cache_valid(self, availability: ProviderSymbolAvailability, now: datetime) -> bool:
        """Cache is valid if 'now' is strictly before 'next_check_at'."""
        return now < availability.next_check_at

    def calculate_next_check(self, status: ProviderSymbolAvailabilityStatus, checked_at: datetime) -> datetime:
        if status == ProviderSymbolAvailabilityStatus.SUPPORTED:
            return checked_at + timedelta(hours=self.availability_ttl_hours)
        elif status == ProviderSymbolAvailabilityStatus.UNSUPPORTED:
            return checked_at + timedelta(hours=self.unsupported_recheck_hours)
        elif status == ProviderSymbolAvailabilityStatus.TEMPORARY_ERROR:
            return checked_at + timedelta(minutes=self.temporary_error_recheck_minutes)
        else:
            raise ValueError(f"Unknown status: {status!r}")

    def create_availability(
        self,
        *,
        source: MarketDataSource,
        requested_symbol: str,
        status: ProviderSymbolAvailabilityStatus,
        now: datetime,
        provider_symbol: str | None = None,
        failure_reason: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ProviderSymbolAvailability:
        next_check = self.calculate_next_check(status, now)
        return ProviderSymbolAvailability(
            source=source,
            requested_symbol=requested_symbol,
            provider_symbol=provider_symbol,
            status=status,
            next_check_at=next_check,
            first_seen_at=now,
            last_checked_at=now,
            failure_reason=failure_reason,
            metadata=metadata,
        )
