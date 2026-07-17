from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage12GapsScanSmokeTests(unittest.TestCase):
    def test_gaps_scan_command_is_read_only_by_default(self) -> None:
        main = (SERVICE_ROOT / "src/market_data_service/main.py").read_text(encoding="utf-8")
        service = (SERVICE_ROOT / "src/market_data_service/application/services/gap_scan_service.py").read_text(
            encoding="utf-8"
        )
        maintenance = (SERVICE_ROOT / "src/market_data_service/runtime/maintenance.py").read_text(encoding="utf-8")

        self.assertIn("gaps:scan", main)
        self.assertIn("--create-backfill", main)
        self.assertIn("Gaps scan report", main)
        self.assertIn("if not command.create_backfill", service)
        self.assertIn("request_backfill(event)", service)
        self.assertIn("run_gaps_scan", maintenance)
        self.assertNotIn("run_http_server", maintenance)

    def test_gaps_scan_reads_candles_through_repository(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/gap_scan_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class GapScanRepository", repository)
        self.assertIn("market_candles.c.open_time", repository)
        self.assertIn("market_candles.c.is_closed.is_(True)", repository)
