from __future__ import annotations

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = SERVICE_ROOT / "docs" / "production_readiness.md"


def test_production_readiness_doc_exists() -> None:
    assert DOC_PATH.exists()


def test_production_readiness_doc_covers_required_stage_18_topics() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    required_phrases = (
        "standard runtime",
        "Operator Runbook",
        "Rollback Plan",
        "Backup/Restore State",
        "Security Review",
        "Load Tests",
        "Failure/Recovery Tests",
        "external rollback",
        "platform-native `spot_grid`",
        "platform-native `spot_greenwich`",
        "bot_platform_service.trading_bots.spot_grid.adapter",
        "bot_platform_service.trading_bots.spot_greenwich.adapter",
        "New production module IDs must be suffix-free",
        "package-local adapter under `trading_bots/<module_id>/adapter.py`",
        "BotModuleContractHarness",
        "dry_run",
        "notification_only",
        "signal_only",
        "private exchange clients",
        "open positions",
        "create exchange orders",
        "duplicate runs",
        "duplicate signals",
        "_bot_platform.bot_runtime_state",
        "SecretProvider",
        "failed bot instance is isolated",
        "supported production runtime or extension path",
    )

    assert [phrase for phrase in required_phrases if phrase not in text] == []


def test_production_readiness_doc_names_verification_commands() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "test_stage_18_production_readiness_documentation.py" in text
    assert "test_stage_18_production_readiness_runtime.py" in text
    assert "test_stage_14_observability_baseline.py" in text
    assert "test_stage_15_bot_module_contract_harness.py" in text
    assert "test_stage_17_migration_rollout.py" in text
    assert "bot-platform-service/tests" in text
