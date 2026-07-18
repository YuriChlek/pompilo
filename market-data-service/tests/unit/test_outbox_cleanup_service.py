from __future__ import annotations

from datetime import UTC, datetime, timedelta
import unittest
from unittest.mock import AsyncMock, MagicMock

from market_data_service.domain.enums import OutboxStatus
from market_data_service.domain.outbox_models import OutboxEvent
from market_data_service.observability.metrics import MARKET_DATA_OUTBOX_EVENTS_DELETED_TOTAL, InMemoryMetricsRecorder
from market_data_service.persistence.repositories.outbox_repository import OutboxRepository
from market_data_service.application.services.outbox_cleanup_service import OutboxCleanupService


class OutboxCleanupServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
        self.repository = MagicMock(spec=OutboxRepository)
        self.metrics = InMemoryMetricsRecorder()
        self.service = OutboxCleanupService(
            outbox_repository=self.repository,
            retention_days=5,
            metrics_recorder=self.metrics,
            now_provider=lambda: self.now,
        )

    async def test_cleanup_calculates_correct_cutoff_and_calls_repository(self) -> None:
        self.repository.delete_old_published_events = AsyncMock(return_value=5)

        deleted = await self.service.cleanup(batch_size=100)

        self.assertEqual(deleted, 5)
        self.assertEqual(self.metrics.samples[0].name, MARKET_DATA_OUTBOX_EVENTS_DELETED_TOTAL)
        self.assertEqual(self.metrics.samples[0].value, 5.0)
        expected_cutoff = self.now - timedelta(days=5)
        self.repository.delete_old_published_events.assert_called_once_with(
            cutoff=expected_cutoff,
            batch_size=100,
        )

    async def test_cleanup_returns_zero_when_no_events_deleted(self) -> None:
        self.repository.delete_old_published_events = AsyncMock(return_value=0)

        deleted = await self.service.cleanup()

        self.assertEqual(deleted, 0)
        self.assertEqual(self.metrics.samples[0].value, 0.0)
        self.repository.delete_old_published_events.assert_called_once()

    async def test_cleanup_rejects_invalid_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            await self.service.cleanup(batch_size=0)
