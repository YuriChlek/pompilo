from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage7OutboxLifecycleSmokeTests(unittest.TestCase):
    def test_outbox_lifecycle_contract_is_wired_into_runtime(self) -> None:
        adapter = (
            SERVICE_ROOT / "src/market_data_service/workers/outbox_publisher_lifecycle.py"
        ).read_text(encoding="utf-8")
        settings = (SERVICE_ROOT / "src/market_data_service/config/settings.py").read_text(encoding="utf-8")
        http_server = (SERVICE_ROOT / "src/market_data_service/runtime/http_server.py").read_text(encoding="utf-8")
        health = (SERVICE_ROOT / "src/market_data_service/runtime/health.py").read_text(encoding="utf-8")
        metrics = (SERVICE_ROOT / "src/market_data_service/observability/metrics.py").read_text(encoding="utf-8")

        for method_name in ("start", "stop", "run_forever", "health"):
            self.assertIn(f"def {method_name}", adapter)

        self.assertIn("except Exception", adapter)
        self.assertIn("record_outbox_publish_batch", adapter)
        self.assertIn("market_data.outbox.lifecycle", adapter)
        self.assertIn("MARKET_DATA_OUTBOX_PUBLISHER_ENABLED", settings)
        self.assertIn("outbox_publisher_lifecycle.start()", http_server)
        self.assertIn("outbox_publisher_lifecycle.stop()", http_server)
        self.assertIn("_check_outbox_publisher", health)
        self.assertIn("market_data_outbox_events_published_total", metrics)
