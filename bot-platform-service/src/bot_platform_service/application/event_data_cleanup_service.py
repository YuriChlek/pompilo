from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol


class EventDataCleanupRepository(Protocol):
    """Persistence boundary for service-owned event cleanup."""

    async def delete_terminal_processed_events_before(self, *, cutoff: datetime, batch_size: int) -> int:
        """Delete old terminal processed event records in a bounded batch."""

    async def delete_market_data_event_audit_records_before(self, *, cutoff: datetime, batch_size: int) -> int:
        """Delete old raw market-data event audit records in a bounded batch."""


@dataclass(frozen=True, slots=True)
class EventDataCleanupResult:
    """Summary of one Bot Platform event-data cleanup batch."""

    processed_event_deleted_count: int
    event_audit_deleted_count: int


class EventDataCleanupService:
    """Cleanup service-owned event records without touching business history."""

    def __init__(
        self,
        *,
        repository: EventDataCleanupRepository,
        idempotency_retention_days: int,
        audit_retention_days: int,
        batch_size: int = 1000,
    ) -> None:
        self.repository = repository
        self.idempotency_retention_days = idempotency_retention_days
        self.audit_retention_days = audit_retention_days
        self.batch_size = batch_size

    async def run_once(self, *, now: datetime | None = None) -> EventDataCleanupResult:
        """Run one idempotent cleanup batch for terminal service event records."""

        current_time = (now or datetime.now(UTC)).astimezone(UTC)
        processed_cutoff = current_time - timedelta(days=self.idempotency_retention_days)
        processed_deleted_count = await self.repository.delete_terminal_processed_events_before(
            cutoff=processed_cutoff,
            batch_size=self.batch_size,
        )
        audit_cutoff = current_time - timedelta(days=self.audit_retention_days)
        audit_deleted_count = await self.repository.delete_market_data_event_audit_records_before(
            cutoff=audit_cutoff,
            batch_size=self.batch_size,
        )
        return EventDataCleanupResult(
            processed_event_deleted_count=processed_deleted_count,
            event_audit_deleted_count=audit_deleted_count,
        )
