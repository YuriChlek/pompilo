from __future__ import annotations

from pathlib import Path
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[2]


class PersistentBackendStage14SnapshotContractSmokeTests(unittest.TestCase):
    def test_snapshot_contract_is_versioned_and_documented(self) -> None:
        service = (
            SERVICE_ROOT / "src/market_data_service/application/services/snapshot_read_service.py"
        ).read_text(encoding="utf-8")
        document = (SERVICE_ROOT / "docs/snapshot_contract_v1.md").read_text(encoding="utf-8")

        self.assertIn('SNAPSHOT_CONTRACT_VERSION = "market-snapshot.v1"', service)
        self.assertIn("GET /snapshots/latest", document)
        self.assertIn("market-snapshot.v1", document)
        self.assertIn("Breaking response changes require a new endpoint or contract version", document)
        self.assertIn("Bot Platform must not read internal", document)

    def test_http_server_exposes_latest_snapshot_contract_route(self) -> None:
        http_server = (SERVICE_ROOT / "src/market_data_service/runtime/http_server.py").read_text(encoding="utf-8")
        repository = (
            SERVICE_ROOT / "src/market_data_service/persistence/repositories/snapshot_repository.py"
        ).read_text(encoding="utf-8")

        self.assertIn("/snapshots/latest", http_server)
        self.assertIn("max_age_seconds", http_server)
        self.assertIn("get_latest_complete_snapshot", repository)
        self.assertIn("CandleRangeStatus.COMPLETE", repository)
