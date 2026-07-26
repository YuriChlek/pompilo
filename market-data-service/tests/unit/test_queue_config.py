from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from market_data_service.config.queue_config import get_redis_stream_broker_config


class QueueConfigTests(unittest.TestCase):
    def test_redis_stream_config_uses_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = get_redis_stream_broker_config()

        self.assertEqual(config.redis_url, "redis://localhost:6379/0")
        self.assertEqual(config.stream_name, "market-data-events")
        self.assertEqual(config.maxlen, 100000)

    def test_redis_stream_config_reads_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "MARKET_DATA_REDIS_URL": "redis://redis:6379/2",
                "MARKET_DATA_OUTBOX_STREAM": "custom-stream",
                "MARKET_DATA_OUTBOX_STREAM_MAXLEN": "10000",
            },
            clear=True,
        ):
            config = get_redis_stream_broker_config()

        self.assertEqual(config.redis_url, "redis://redis:6379/2")
        self.assertEqual(config.stream_name, "custom-stream")
        self.assertEqual(config.maxlen, 10000)

    def test_redis_stream_config_derives_maxlen_from_retention_days(self) -> None:
        with patch.dict(os.environ, {"MARKET_DATA_REDIS_EVENT_RETENTION_DAYS": "3"}, clear=True):
            config = get_redis_stream_broker_config()

        self.assertEqual(config.maxlen, 150000)

    def test_redis_stream_config_rejects_invalid_maxlen_values(self) -> None:
        with patch.dict(os.environ, {"MARKET_DATA_OUTBOX_STREAM_MAXLEN": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MARKET_DATA_OUTBOX_STREAM_MAXLEN"):
                get_redis_stream_broker_config()
