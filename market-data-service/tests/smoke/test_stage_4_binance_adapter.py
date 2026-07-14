from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage4BinanceAdapterSmokeTests(unittest.TestCase):
    def test_binance_adapter_is_read_only_and_provider_scoped(self) -> None:
        adapter = (
            SERVICE_ROOT / "src/market_data_service/infrastructure/providers/binance_spot_adapter.py"
        ).read_text(encoding="utf-8")

        self.assertIn("async def fetch_closed_candles", adapter)
        self.assertIn("ProviderSymbol", adapter)
        self.assertIn("normalize_closed_candle", adapter)
        self.assertIn("validate_closed_candle", adapter)
        self.assertIn("RetryableProviderError", adapter)
        self.assertIn("NonRetryableProviderError", adapter)
        self.assertNotIn("sqlalchemy", adapter.lower())
        self.assertNotIn("persistence", adapter.lower())
        self.assertNotIn("strategy", adapter.lower())

    def test_pyproject_declares_httpx_for_provider_transport(self) -> None:
        pyproject = (SERVICE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"httpx>=0.27"', pyproject)
