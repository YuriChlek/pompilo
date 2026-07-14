from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage11OutboxPublisherSmokeTests(unittest.TestCase):
    def test_outbox_publisher_service_has_retry_backoff_and_lag_metric(self) -> None:
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/outbox_publisher_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class OutboxPublisherService", service)
        self.assertIn("max_attempts", service)
        self.assertIn("retry_backoff_seconds", service)
        self.assertIn("mark_published", service)
        self.assertIn("mark_retry", service)
        self.assertIn("mark_failed", service)
        self.assertIn("publisher_lag_seconds", service)

    def test_outbox_repository_reads_and_updates_outbox_rows(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/outbox_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("fetch_publishable_events", repository)
        self.assertIn("mark_published", repository)
        self.assertIn("mark_retry", repository)
        self.assertIn("mark_failed", repository)
        self.assertIn("OutboxStatus.PUBLISHED", repository)
        self.assertIn("OutboxStatus.FAILED", repository)

    def test_outbox_publisher_worker_is_thin_entrypoint(self) -> None:
        worker = (
            SERVICE_ROOT / "src/market_data_service/workers/outbox_publisher_worker.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class OutboxPublisherWorker", worker)
        self.assertIn("publish_once", worker)
        self.assertIn("run_forever", worker)
        self.assertNotIn("sqlalchemy", worker.lower())
        self.assertNotIn("select(", worker)

    def test_redis_stream_broker_adapter_exists_for_production_delivery(self) -> None:
        adapter = (
            SERVICE_ROOT / "src/market_data_service/infrastructure/queues/redis_stream_broker.py"
        ).read_text(encoding="utf-8")
        pyproject = (SERVICE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("class RedisStreamEventBroker", adapter)
        self.assertIn("async def publish", adapter)
        self.assertIn("xadd", adapter)
        self.assertIn('"redis>=5.0"', pyproject)
