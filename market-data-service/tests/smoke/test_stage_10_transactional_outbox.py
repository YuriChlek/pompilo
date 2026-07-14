from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage10TransactionalOutboxSmokeTests(unittest.TestCase):
    def test_outbox_schema_and_repository_exist(self) -> None:
        tables = (SERVICE_ROOT / "src/market_data_service/persistence/tables/outbox_events_tables.py").read_text(
            encoding="utf-8"
        )
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/outbox_repository.py"
        ).read_text(encoding="utf-8")
        migration = (SERVICE_ROOT / "alembic/versions/20260714_0006_create_outbox_events.py").read_text(
            encoding="utf-8"
        )

        for field_name in (
            "event_type",
            "aggregate_type",
            "aggregate_id",
            "payload_json",
            "idempotency_key",
            "status",
            "attempts",
            "next_attempt_at",
            "published_at",
        ):
            self.assertIn(field_name, tables)
            self.assertIn(field_name, migration)

        self.assertIn("outbox_events_event_type_idempotency_key_uq", migration)
        self.assertIn("create_pending_candle_batch_ready", repository)
        self.assertIn("on_conflict_do_nothing", repository)
        self.assertIn("OutboxStatus.PENDING", repository)

    def test_complete_sync_writes_outbox_without_direct_publish(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/sync_completion_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("create_pending_candle_batch_ready", repository)
        self.assertIn("CandleBatchReady.from_snapshot", repository)
        self.assertNotIn("publish", repository.lower())
        self.assertNotIn("redis", repository.lower())
        self.assertNotIn("rabbit", repository.lower())

    def test_candle_batch_ready_event_contract_lives_in_domain_events(self) -> None:
        event_contract = (
            SERVICE_ROOT / "src/market_data_service/domain/events/candle_batch_ready.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class CandleBatchReady", event_contract)
        self.assertIn("event_id", event_contract)
        self.assertIn("occurred_at", event_contract)
        self.assertIn("idempotency_key", event_contract)
        self.assertIn("snapshot_version", event_contract)
        self.assertNotIn("sqlalchemy", event_contract.lower())
