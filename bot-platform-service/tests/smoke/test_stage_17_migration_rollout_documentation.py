from __future__ import annotations

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = SERVICE_ROOT / "docs" / "migration_rollout.md"


def test_migration_rollout_doc_exists() -> None:
    assert DOC_PATH.exists()


def test_migration_rollout_doc_covers_required_rollout_gates() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    required = (
        "spot_grid",
        "dry_run",
        "Zero-Persistence Preview",
        "Platform `dry_run` is not this preview path",
        "spot_greenwich",
        "notification_only",
        "Standalone Comparison",
        "signal_only",
        "Rollback",
        "Standalone CLI",
        "must not open positions",
        "private exchange client",
        "candle sync",
        "at least 10 stable comparable runs",
    )

    assert [phrase for phrase in required if phrase not in text] == []


def test_migration_rollout_doc_names_test_commands() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "test_stage_17_migration_rollout.py" in text
    assert "bot-platform-service/tests" in text


def test_migration_rollout_doc_does_not_claim_dry_run_is_always_zero_persistence() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")

    assert "dry_run always" not in text
    assert "dry_run` always" not in text
    assert "Platform `dry_run` is not this preview path" in text
    assert "may have persistence side effects" in text
