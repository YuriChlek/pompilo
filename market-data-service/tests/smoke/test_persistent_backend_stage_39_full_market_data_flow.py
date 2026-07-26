from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage39FullMarketDataFlowSmokeTests(unittest.TestCase):
    def test_docker_runtime_smoke_covers_full_market_data_flow(self) -> None:
        script = (SERVICE_ROOT / "scripts/docker_runtime_smoke.sh").read_text(encoding="utf-8")

        for required_text in (
            "MARKET_DATA_PROVIDER_MODE=fixture",
            "market_data_migrate",
            "reset_market_data_smoke_state",
            "collect --once",
            "bootstrap market_candles",
            "bootstrap market_snapshots",
            "bootstrap outbox_events",
            "bootstrap Redis stream",
            "prepare_incremental_gap",
            "incremental published outbox_events",
            "incremental Redis stream",
            "outbox:cleanup",
            "market_candles after outbox cleanup",
            "mark_collection_current_for_idempotent_rerun",
            "market_candles after idempotent rerun",
            "duplicate candle natural keys",
            "Market Data full flow smoke passed.",
        ):
            self.assertIn(required_text, script)

    def test_docker_runtime_smoke_uses_isolated_fixture_stream(self) -> None:
        script = (SERVICE_ROOT / "scripts/docker_runtime_smoke.sh").read_text(encoding="utf-8")

        self.assertIn("market-data-events-smoke", script)
        self.assertIn("redis_scalar DEL \"$OUTBOX_STREAM\"", script)
        self.assertNotIn("symbols:sync", script)
        self.assertNotIn("scheduler:run-once", script)
        self.assertNotIn("sync:run-next", script)
        self.assertNotIn("outbox:publish-once", script)
