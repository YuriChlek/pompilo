from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class Stage9CompleteBatchSnapshotTransactionSmokeTests(unittest.TestCase):
    def test_sync_completion_repository_uses_one_transaction_boundary(self) -> None:
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/sync_completion_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("class SyncCompletionRepository", repository)
        self.assertIn("_transaction_boundary", repository)
        self.assertIn("insert_closed_candles", repository)
        self.assertIn("complete_batch", repository)
        self.assertIn("create_snapshot_if_changed", repository)
        self.assertIn("MarketDataBatchStatus.COMPLETE", repository)

    def test_single_symbol_sync_can_return_snapshot_identity_without_strategy_execution(self) -> None:
        sync_models = (SERVICE_ROOT / "src/market_data_service/application/sync_models.py").read_text(
            encoding="utf-8"
        )
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/single_symbol_sync_service.py"
        ).read_text(encoding="utf-8")

        self.assertIn("snapshot_id", sync_models)
        self.assertIn("snapshot_created", sync_models)
        self.assertIn("SyncCompletionPort", service)
        self.assertNotIn("StrategyMarketDataReady", service)
        self.assertNotIn("CandleBatchReady", service)
