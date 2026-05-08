from __future__ import annotations

from pathlib import Path
import unittest


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
        self.assertIn("HEALTHCHECK_PORT=8080", content)
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
