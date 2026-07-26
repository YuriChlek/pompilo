from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage7BatchTrackingSmokeTests(unittest.TestCase):
    def test_batch_tracking_schema_and_repository_exist(self) -> None:
        tables = (SERVICE_ROOT / "src/market_data_service/persistence/tables/market_data_batches_tables.py").read_text(
            encoding="utf-8"
        )
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/batch_repository.py"
        ).read_text(encoding="utf-8")
        migration = (SERVICE_ROOT / "alembic/versions/20260714_0004_create_market_data_batches.py").read_text(
            encoding="utf-8"
        )

        for field_name in (
            "batch_id",
            "status",
            "outbox_status",
            "rows_fetched",
            "rows_inserted",
            "rows_skipped_duplicate",
            "rows_hash_mismatch",
            "gap_count",
            "error_code",
            "completed_at",
        ):
            self.assertIn(field_name, tables)
            self.assertIn(field_name, migration)

        self.assertIn("create_running_batch", repository)
        self.assertIn("complete_batch", repository)
        self.assertIn("fail_batch", repository)
        self.assertIn("NOT_CREATED", repository)

    def test_sync_service_uses_batch_tracker_without_event_publishing(self) -> None:
        ports = (SERVICE_ROOT / "src/market_data_service/application/market_data_ports.py").read_text(encoding="utf-8")
        models = (SERVICE_ROOT / "src/market_data_service/application/sync_models.py").read_text(encoding="utf-8")
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/single_symbol_sync_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("BatchTrackerPort", ports)
        self.assertIn("batch_id", models)
        self.assertIn("batch_status", models)
        self.assertIn("create_running_batch", service)
        self.assertIn("complete_batch", service)
        self.assertIn("fail_batch", service)
        self.assertNotIn("StrategyMarketDataReady", service)
        self.assertNotIn("CandleBatchReady", service)
