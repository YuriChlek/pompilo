from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage5LifecycleSmokeTests(unittest.TestCase):
    def test_lifecycle_and_signal_shutdown_contract_exist(self) -> None:
        lifecycle = (SERVICE_ROOT / "src/market_data_service/runtime/lifecycle.py").read_text(encoding="utf-8")
        signals = (SERVICE_ROOT / "src/market_data_service/runtime/signals.py").read_text(encoding="utf-8")
        http_server = (SERVICE_ROOT / "src/market_data_service/runtime/http_server.py").read_text(encoding="utf-8")
        health = (SERVICE_ROOT / "src/market_data_service/runtime/health.py").read_text(encoding="utf-8")

        for state in ("starting", "ready", "shutting_down"):
            self.assertIn(state, lifecycle)

        self.assertIn("SIGTERM", signals)
        self.assertIn("SIGINT", signals)
        self.assertIn("wait_for_shutdown", http_server)
        self.assertIn("settings.shutdown.timeout_seconds", http_server)
        self.assertIn("is_shutting_down", health)
