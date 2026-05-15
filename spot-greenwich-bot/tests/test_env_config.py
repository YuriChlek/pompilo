from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from utils.env import load_project_env


class EnvConfigTests(unittest.TestCase):
    def test_env_documents_required_vars(self) -> None:
        content = (Path(__file__).resolve().parent.parent / ".env").read_text(encoding="utf-8")

        self.assertIn("DB_HOST=localhost", content)
        self.assertIn("BYBIT_API_KEY=", content)
        self.assertIn("BINANCE_REST_ENDPOINT=https://api.binance.com", content)
        self.assertIn("ORDER_DEPOSIT_PERCENT=5", content)
        self.assertIn("AVERAGING_ENTRY_LIMIT=3", content)
        self.assertIn("AVERAGING_ENTRY_2_SIZE_PERCENT=60", content)
        self.assertIn("AVERAGING_ENTRY_3_SIZE_PERCENT=30", content)
        self.assertIn("MIN_PROFIT_RATIO=0.01", content)
        self.assertIn("NO_LOSS_GUARD_ENABLED=true", content)
        self.assertIn("TELEGRAM_BOT_TOKEN=", content)
        self.assertIn("ATR_POSITION_SIZING_ENABLED=true", content)
        self.assertIn("ATR_POSITION_SIZING_MEDIAN_WINDOW=50", content)
        self.assertIn("ATR_POSITION_SIZING_MIN_MULTIPLIER=0.5", content)
        self.assertIn("ATR_POSITION_SIZING_MAX_MULTIPLIER=1.5", content)
        self.assertIn("BUY_PRICE_GUARD_ENABLED=true", content)
        self.assertIn("BUY_PRICE_GUARD_MAX_DEVIATION_RATIO=0.01", content)
        self.assertIn("PORTFOLIO_CAP_ENABLED=true", content)
        self.assertIn("PORTFOLIO_POSITION_LIMIT=3", content)
        self.assertIn("PORTFOLIO_PRIORITY_SYMBOLS=BTCUSDT,ETHUSDT", content)
        self.assertIn("DRY_RUN_QUOTE_BALANCE=1000", content)

    def test_load_project_env_uses_production_first_and_env_as_fallback(self) -> None:
        with TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            (project_root / ".env.production").write_text("PRIMARY_ONLY=1\nSHARED=prod\n", encoding="utf-8")
            (project_root / ".env").write_text("FALLBACK_ONLY=1\nSHARED=dev\n", encoding="utf-8")

            with patch("utils.env.Path.resolve", return_value=project_root / "utils" / "env.py"):
                with patch.dict(os.environ, {}, clear=True):
                    loaded_paths = load_project_env()
                    self.assertEqual(os.environ["PRIMARY_ONLY"], "1")
                    self.assertEqual(os.environ["FALLBACK_ONLY"], "1")
                    self.assertEqual(os.environ["SHARED"], "prod")

        self.assertEqual(loaded_paths, (project_root / ".env.production", project_root / ".env"))
