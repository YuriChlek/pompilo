from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging

from market_data_service.observability.metrics import MetricsRecorder, record_outbox_cleanup
from market_data_service.observability.structured_logging import StructuredLogger
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository

logger = logging.getLogger(__name__)


class OutboxCleanupService:
    def __init__(
        self,
        *,
        outbox_repository: OutboxRepository,
        retention_days: int,
        metrics_recorder: MetricsRecorder | None = None,
        structured_logger: StructuredLogger | None = None,
        now_provider=None,
    ) -> None:
        self.outbox_repository = outbox_repository
        self.retention_days = retention_days
        self.metrics_recorder = metrics_recorder
        self.structured_logger = structured_logger
        self.now_provider = now_provider or (lambda: datetime.now(UTC))

    async def cleanup(self, *, batch_size: int = 1000) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        now = self.now_provider()
        cutoff = now - timedelta(days=self.retention_days)

        deleted_count = await self.outbox_repository.delete_old_published_events(
            cutoff=cutoff,
            batch_size=batch_size,
        )

        record_outbox_cleanup(self.metrics_recorder, deleted_count=deleted_count)

        if deleted_count > 0:
            logger.info("Outbox cleanup deleted %d old published events (cutoff: %s)", deleted_count, cutoff.isoformat())
            if self.structured_logger is not None:
                self.structured_logger.emit({
                    "event_type": "outbox_cleanup",
                    "deleted_count": deleted_count,
                    "cutoff": cutoff.isoformat(),
                })
        return deleted_count
