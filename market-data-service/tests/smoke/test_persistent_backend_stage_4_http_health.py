from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage4HttpHealthSmokeTests(unittest.TestCase):
    def test_http_health_runtime_and_cli_contract_exist(self) -> None:
        http_server = (SERVICE_ROOT / "src/market_data_service/runtime/http_server.py").read_text(encoding="utf-8")
        health = (SERVICE_ROOT / "src/market_data_service/runtime/health.py").read_text(encoding="utf-8")
        main = (SERVICE_ROOT / "src/market_data_service/main.py").read_text(encoding="utf-8")

        for route in ("/health/live", "/health/ready", "/metrics"):
            self.assertIn(route, http_server)

        self.assertIn("select 1", health)
        self.assertIn("ping", health)
        self.assertIn("market_data_alembic_version", health)
        self.assertIn("run_http_server", main)
        self.assertIn("http://127.0.0.1:", main)
        self.assertIn("/health/ready", main)
