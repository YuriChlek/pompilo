from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage2SettingsLayerSmokeTests(unittest.TestCase):
    def test_typed_settings_layer_exists_for_required_groups(self) -> None:
        settings = (SERVICE_ROOT / "src/market_data_service/config/settings.py").read_text(encoding="utf-8")

        for class_name in (
            "DatabaseSettings",
            "RedisStreamBrokerConfig",
            "BinanceSpotProviderConfig",
            "SchedulerConfig",
            "HttpSettings",
            "LoggingSettings",
            "ShutdownSettings",
            "MarketDataServiceSettings",
        ):
            self.assertIn(class_name, settings)

        self.assertIn("MARKET_DATA_DATABASE_URL", settings)
        self.assertIn("DB_NAME", settings)
        self.assertIn("MARKET_DATA_REDIS_URL", settings)
        self.assertIn("SettingsError", settings)

    def test_migration_job_env_names_remain_unchanged(self) -> None:
        compose = (SERVICE_ROOT.parent / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

        self.assertIn("DB_HOST: postgres", compose)
        self.assertIn("DB_NAME: ${DB_NAME:-pampilo_db}", compose)
        self.assertIn("DATABASE: ${DB_NAME:-pampilo_db}", compose)
        self.assertIn('command: ["alembic", "upgrade", "head"]', compose)
