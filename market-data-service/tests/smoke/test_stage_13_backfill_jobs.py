from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage13BackfillJobsSmokeTests(unittest.TestCase):
    def test_backfill_event_and_planner_exist(self) -> None:
        event = (
            SERVICE_ROOT / "src/market_data_service/domain/events/market_data_backfill_requested.py"
        ).read_text(encoding="utf-8")
        planner = (
            SERVICE_ROOT / "src/market_data_service/application/services/backfill_planning_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("MarketDataBackfillRequested", event)
        self.assertIn("GAP_DETECTED", event)
        self.assertIn("BackfillPlanningService", planner)
        self.assertIn("request_backfill_for_gaps", planner)

    def test_sync_jobs_support_backfill_priority_and_ranges(self) -> None:
        table = (SERVICE_ROOT / "src/market_data_service/persistence/tables/sync_jobs_tables.py").read_text(
            encoding="utf-8"
        )
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/sync_job_repository.py"
        ).read_text(encoding="utf-8")
        migration = (
            SERVICE_ROOT / "alembic/versions/20260714_0008_extend_sync_jobs_for_backfill.py"
        ).read_text(encoding="utf-8")

        self.assertIn("job_kind", table)
        self.assertIn("priority", table)
        self.assertIn("requested_from", table)
        self.assertIn("requested_to", table)
        self.assertIn("enqueue_backfill_job", repository)
        self.assertIn("order_by(sync_jobs.c.priority", repository)
        self.assertIn("BACKFILL", migration)

    def test_sync_service_routes_gap_detected_to_backfill_without_ready_for_incomplete(self) -> None:
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/single_symbol_sync_service.py"
        ).read_text(encoding="utf-8")
        completion = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/sync_completion_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("CandleRangeStatus.GAP_DETECTED", service)
        self.assertIn("request_backfill_for_gaps", service)
        self.assertIn("batch_status == MarketDataBatchStatus.COMPLETE", completion)
        self.assertIn("create_pending_candle_batch_ready", completion)
