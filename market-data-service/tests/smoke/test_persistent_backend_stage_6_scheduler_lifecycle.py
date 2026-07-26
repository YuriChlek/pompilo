from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage6SchedulerLifecycleSmokeTests(unittest.TestCase):
    def test_scheduler_lifecycle_contract_is_wired_into_runtime(self) -> None:
        adapter = (SERVICE_ROOT / "src/market_data_service/workers/scheduler_lifecycle.py").read_text(encoding="utf-8")
        settings = (SERVICE_ROOT / "src/market_data_service/config/settings.py").read_text(encoding="utf-8")
        http_server = (SERVICE_ROOT / "src/market_data_service/runtime/http_server.py").read_text(encoding="utf-8")
        health = (SERVICE_ROOT / "src/market_data_service/runtime/health.py").read_text(encoding="utf-8")

        for method_name in ("start", "stop", "run_forever", "health"):
            self.assertIn(f"def {method_name}", adapter)

        self.assertIn("except Exception", adapter)
        self.assertIn("record_scheduler_tick", adapter)
        self.assertIn("market_data.scheduler.tick", adapter)
        self.assertIn("MARKET_DATA_SCHEDULER_ENABLED", settings)
        self.assertIn("scheduler_lifecycle.start()", http_server)
        self.assertIn("scheduler_lifecycle.stop()", http_server)
        self.assertIn("_check_scheduler", health)
