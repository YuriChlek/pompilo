from __future__ import annotations

from pathlib import Path
import unittest


class EnvExampleTests(unittest.TestCase):
    def test_env_example_documents_production_precedence_and_required_vars(self) -> None:
        content = (Path(__file__).resolve().parent.parent / ".env.example").read_text(encoding="utf-8")

        self.assertIn(".env.production", content)
        self.assertIn(".env", content)
        self.assertIn("BYBIT_API_KEY=", content)
        self.assertIn("BINANCE_REST_ENDPOINT=https://api.binance.com", content)
        self.assertIn("ORDER_DEPOSIT_PERCENT=5", content)
        self.assertIn("MIN_PROFIT_RATIO=0.01", content)
        self.assertIn("NO_LOSS_GUARD_ENABLED=true", content)
        self.assertIn("TELEGRAM_BOT_TOKEN=", content)
        self.assertIn("HEALTHCHECK_PORT=8080", content)
