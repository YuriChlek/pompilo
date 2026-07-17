from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from market_data_service.config.database_config import get_database_url
from market_data_service.config.settings import SettingsError, load_database_settings, load_settings


class MarketDataServiceSettingsTests(unittest.TestCase):
    def test_settings_use_local_defaults(self) -> None:
        settings = load_settings({})

        self.assertEqual(
            settings.database.database_url,
            "postgresql+asyncpg://admin:admin_pass@localhost:5432/pompilo_db",
        )
        self.assertEqual(settings.redis.redis_url, "redis://localhost:6379/0")
        self.assertEqual(settings.redis.stream_name, "market-data-events")
        self.assertEqual(settings.provider.rest_endpoint, "https://api.binance.com")
        self.assertEqual(settings.provider_mode, "binance")
        self.assertEqual(settings.scheduler.provider_symbols, ("ETHUSDT",))
        self.assertEqual(settings.scheduler.timeframes, ("1h", "4h", "1d"))
        self.assertEqual(settings.http.host, "0.0.0.0")
        self.assertEqual(settings.http.port, 8010)
        self.assertEqual(settings.logging.level, "INFO")
        self.assertEqual(settings.shutdown.timeout_seconds, 30.0)
        self.assertEqual(settings.backfill.batch_size_candles, 500)
        self.assertEqual(settings.backfill.max_concurrency, 1)
        self.assertTrue(settings.scheduler_enabled)
        self.assertTrue(settings.outbox_publisher_enabled)

    def test_settings_keep_docker_env_compatibility(self) -> None:
        settings = load_settings(
            {
                "DB_HOST": "postgres",
                "DB_PORT": "5432",
                "DB_USER": "admin",
                "DB_PASSWORD": "admin_pass",
                "DB_NAME": "pampilo_db",
                "MARKET_DATA_REDIS_URL": "redis://redis:6379/0",
                "MARKET_DATA_PROVIDER_SYMBOLS": "btcusdt, ethusdt",
                "MARKET_DATA_TIMEFRAMES": "1h,4h",
                "MARKET_DATA_HTTP_HOST": "127.0.0.1",
                "MARKET_DATA_HTTP_PORT": "8011",
                "MARKET_DATA_LOG_LEVEL": "debug",
                "MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS": "15",
                "MARKET_DATA_BACKFILL_BATCH_CANDLES": "250",
                "MARKET_DATA_BACKFILL_MAX_CONCURRENCY": "2",
                "MARKET_DATA_PROVIDER_MODE": "fixture",
                "MARKET_DATA_SCHEDULER_ENABLED": "false",
                "MARKET_DATA_OUTBOX_PUBLISHER_ENABLED": "false",
            }
        )

        self.assertEqual(
            settings.database.database_url,
            "postgresql+asyncpg://admin:admin_pass@postgres:5432/pampilo_db",
        )
        self.assertEqual(settings.redis.redis_url, "redis://redis:6379/0")
        self.assertEqual(settings.provider_mode, "fixture")
        self.assertEqual(settings.scheduler.provider_symbols, ("BTCUSDT", "ETHUSDT"))
        self.assertEqual(settings.scheduler.timeframes, ("1h", "4h"))
        self.assertEqual(settings.http.host, "127.0.0.1")
        self.assertEqual(settings.http.port, 8011)
        self.assertEqual(settings.logging.level, "DEBUG")
        self.assertEqual(settings.shutdown.timeout_seconds, 15.0)
        self.assertEqual(settings.backfill.batch_size_candles, 250)
        self.assertEqual(settings.backfill.max_concurrency, 2)
        self.assertFalse(settings.scheduler_enabled)
        self.assertFalse(settings.outbox_publisher_enabled)

    def test_market_data_database_url_has_priority(self) -> None:
        settings = load_database_settings(
            {
                "MARKET_DATA_DATABASE_URL": "postgresql+asyncpg://market:secret@db-primary:5432/market_db",
                "DATABASE_URL": "postgresql+asyncpg://legacy:secret@db-legacy:5432/legacy_db",
                "DB_HOST": "ignored",
            }
        )

        self.assertEqual(settings.database_url, "postgresql+asyncpg://market:secret@db-primary:5432/market_db")

    def test_database_url_wrapper_uses_settings_layer_without_changing_env_names(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DB_HOST": "postgres",
                "DB_PORT": "5432",
                "DB_USER": "admin",
                "DB_PASSWORD": "admin_pass",
                "DB_NAME": "pampilo_db",
            },
            clear=True,
        ):
            database_url = get_database_url()

        self.assertEqual(database_url, "postgresql+asyncpg://admin:admin_pass@postgres:5432/pampilo_db")

    def test_invalid_env_values_raise_clear_settings_errors(self) -> None:
        invalid_env_cases = (
            ({"MARKET_DATA_REDIS_URL": "http://redis:6379"}, "MARKET_DATA_REDIS_URL"),
            ({"MARKET_DATA_HTTP_PORT": "70000"}, "MARKET_DATA_HTTP_PORT"),
            ({"MARKET_DATA_LOG_LEVEL": "verbose"}, "MARKET_DATA_LOG_LEVEL"),
            ({"MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS": "0"}, "MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS"),
            ({"MARKET_DATA_TIMEFRAMES": "15m"}, "Unsupported scheduler timeframe"),
            ({"BINANCE_REST_ENDPOINT": "api.binance.com"}, "BINANCE_REST_ENDPOINT"),
            ({"MARKET_DATA_SCHEDULER_ENABLED": "maybe"}, "MARKET_DATA_SCHEDULER_ENABLED"),
            ({"MARKET_DATA_OUTBOX_PUBLISHER_ENABLED": "maybe"}, "MARKET_DATA_OUTBOX_PUBLISHER_ENABLED"),
            ({"MARKET_DATA_BACKFILL_BATCH_CANDLES": "0"}, "MARKET_DATA_BACKFILL_BATCH_CANDLES"),
            ({"MARKET_DATA_BACKFILL_MAX_CONCURRENCY": "0"}, "MARKET_DATA_BACKFILL_MAX_CONCURRENCY"),
            ({"MARKET_DATA_PROVIDER_MODE": "paper"}, "MARKET_DATA_PROVIDER_MODE"),
        )

        for env, expected_message in invalid_env_cases:
            with self.subTest(env=env):
                with self.assertRaisesRegex(SettingsError, expected_message):
                    load_settings(env)
