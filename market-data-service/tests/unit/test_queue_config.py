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
        self.assertIsNone(config.maxlen)

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
