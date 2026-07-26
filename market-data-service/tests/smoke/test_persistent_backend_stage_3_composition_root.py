from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage3CompositionRootSmokeTests(unittest.TestCase):
    def test_runtime_container_module_wires_infrastructure_without_running_loops(self) -> None:
        container = (SERVICE_ROOT / "src/market_data_service/runtime/container.py").read_text(encoding="utf-8")

        for required_name in (
            "build_runtime_container",
            "MarketDataRuntimeContainer",
            "create_async_engine",
            "RedisStreamEventBroker",
            "BinanceSpotAdapter",
            "SyncJobRepository",
            "SingleSymbolSyncService",
            "MarketDataSchedulerWorker",
            "OutboxPublisherWorker",
            "close",
        ):
            self.assertIn(required_name, container)

        self.assertNotIn("run_forever(", container)
        self.assertNotIn("build_sync_jobs(", container)
        self.assertNotIn("detect_candle_range_status(", container)
        self.assertNotIn("normalize_closed_candle(", container)
