from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage12MarketDataSchedulerSmokeTests(unittest.TestCase):
    def test_scheduler_domain_plans_supported_close_boundaries(self) -> None:
        scheduler = (SERVICE_ROOT / "src/market_data_service/domain/scheduler.py").read_text(encoding="utf-8")

        self.assertIn('SUPPORTED_SCHEDULE_TIMEFRAMES = ("1h", "4h", "1d")', scheduler)
        self.assertIn("latest_closed_candle_time", scheduler)
        self.assertIn("due_closed_candle_time", scheduler)
        self.assertIn("safety_delay", scheduler)
        self.assertIn("jitter_seconds", scheduler)

    def test_sync_jobs_have_schema_repository_and_migration(self) -> None:
        table = (SERVICE_ROOT / "src/market_data_service/persistence/tables/sync_jobs_tables.py").read_text(
            encoding="utf-8"
        )
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/sync_job_repository.py"
        ).read_text(encoding="utf-8")
        migration = (SERVICE_ROOT / "alembic/versions/20260714_0007_create_sync_jobs.py").read_text(encoding="utf-8")

        self.assertIn("sync_jobs", table)
        self.assertIn("idempotency_key", table)
        self.assertIn("sync_jobs_logical_sync_uq", table)
        self.assertIn("class SyncJobRepository", repository)
        self.assertIn("on_conflict_do_nothing", repository)
        self.assertIn("create_table", migration)
        self.assertIn("sync_jobs_status_scheduled_for_idx", migration)

    def test_scheduler_worker_is_thin_entrypoint(self) -> None:
        worker = (SERVICE_ROOT / "src/market_data_service/workers/market_data_scheduler_worker.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("class MarketDataSchedulerWorker", worker)
        self.assertIn("run_forever", worker)
        self.assertIn("scheduler_service.tick", worker)
        self.assertNotIn("sqlalchemy", worker.lower())
        self.assertNotIn("select(", worker)
