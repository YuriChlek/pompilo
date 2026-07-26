from __future__ import annotations

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = SERVICE_ROOT / "docs" / "how_to_add_new_bot_module.md"


def test_how_to_add_new_bot_module_doc_exists() -> None:
    assert DOC_PATH.exists()


def test_how_to_add_new_bot_module_doc_covers_required_topics() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    required_phrases = (
        "Recommended Structure",
        "Manifest",
        "Config Schema",
        "BotModule Adapter",
        "Runtime Context",
        "Market Data",
        "Signals",
        "BotRunResult",
        "Config, Secrets, And Permissions",
        "Snapshot Integration",
        "No-Exchange-Client Rule",
        "Idempotency",
        "Contract Tests",
        "Checklist",
        "BotModuleContractHarness",
        "BotSignal.build",
        "SignalPublisher",
        "NotificationPublisher",
        "StateStore",
        "SecretProvider",
        "bot-platform-service/src/bot_platform_service/trading_bots/",
        "bot_platform_service.trading_bots.example.adapter",
        "config_schema.py",
        "module_id` values must not end with `_bot",
        "SQLAlchemy Core table definitions and repositories",
        "Whether dry-run signals are persisted is a platform orchestration decision",
        "persistence depends on the platform entrypoint",
    )

    missing = [phrase for phrase in required_phrases if phrase not in text]
    assert missing == []


def test_how_to_add_new_bot_module_doc_names_test_commands() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "spot_grid_bot/.venv/bin/python -m pytest" in text
    assert "bot-platform-service/tests" in text


def test_how_to_add_new_bot_module_doc_does_not_claim_dry_run_is_zero_persistence() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "dry_run always" not in text
    assert "dry_run` always" not in text
    assert "non-persisting planning pass" not in text
