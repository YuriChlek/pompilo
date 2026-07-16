from __future__ import annotations

import importlib
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from utils import config


class EnvConfigTests(unittest.TestCase):
    def test_env_documents_market_data_service_feature_flag(self) -> None:
        content = (Path(__file__).resolve().parent.parent / ".env").read_text(encoding="utf-8")

        self.assertIn("MARKET_DATA_SERVICE_ENABLED=false", content)

    def test_market_data_service_feature_flag_defaults_to_false(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(config._get_bool_env("MARKET_DATA_SERVICE_ENABLED", "false"))

    def test_market_data_service_feature_flag_accepts_enabled_values(self) -> None:
        enabled_values = ("1", "true", "yes", "on", "TRUE")

        for value in enabled_values:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"MARKET_DATA_SERVICE_ENABLED": value}):
                    self.assertTrue(config._get_bool_env("MARKET_DATA_SERVICE_ENABLED", "false"))

    def test_market_data_service_runtime_constant_is_disabled_by_default(self) -> None:
        with patch.dict(os.environ, {"MARKET_DATA_SERVICE_ENABLED": "false"}):
            reloaded_config = importlib.reload(config)

        self.assertFalse(reloaded_config.MARKET_DATA_SERVICE_ENABLED)
        importlib.reload(config)
