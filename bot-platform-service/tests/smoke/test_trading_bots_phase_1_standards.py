from __future__ import annotations

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
STANDARDS_PATH = SERVICE_ROOT / "STANDARDS.md"
MIGRATION_PLAN_PATH = SERVICE_ROOT / "docs" / "trading_bots_module_migration_plan.md"


def test_phase_1_standards_align_with_trading_bots_migration_plan() -> None:
    standards = STANDARDS_PATH.read_text(encoding="utf-8")
    migration_plan = MIGRATION_PLAN_PATH.read_text(encoding="utf-8")

    required_phrases = (
        "src/bot_platform_service/trading_bots/",
        "trading_bots/<module_id>/",
        "manifest.py",
        "adapter.py",
        "config_schema.py",
        "bot_platform_service.trading_bots.<module_id>.adapter",
        "must not end\nwith `_bot`",
        "Internal infrastructure adapters inside `trading_bots/<module_id>/infrastructure/`",
        "SQLAlchemy Core table definitions and repositories",
        "classic SQLAlchemy ORM",
    )

    assert "Phase 1. Align Standards And Naming" in migration_plan
    assert [phrase for phrase in required_phrases if phrase not in standards] == []


def test_phase_1_standards_retire_legacy_adapter_path() -> None:
    standards = STANDARDS_PATH.read_text(encoding="utf-8")

    assert "bot_platform_service.infrastructure.bot_modules.*" not in standards
    assert "temporary compatibility" not in standards
    assert "service-level\n  infrastructure packages" in standards


def test_phase_1_migration_plan_does_not_claim_dry_run_is_zero_persistence() -> None:
    migration_plan = MIGRATION_PLAN_PATH.read_text(encoding="utf-8")

    assert "dry_run(request)` calculates signals without side effects" not in migration_plan
    assert "Platform persistence is controlled by the orchestration entrypoint" in migration_plan
