from __future__ import annotations

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
RECOVERY_PATH = SERVICE_ROOT / "src" / "bot_platform_service" / "application" / "bot_runtime_recovery_service.py"


def test_runtime_recovery_service_exists() -> None:
    assert RECOVERY_PATH.exists()


def test_runtime_recovery_service_uses_application_protocols() -> None:
    text = RECOVERY_PATH.read_text(encoding="utf-8")
    required = (
        "BotRuntimeRecoveryRunRepository",
        "BotRuntimeRecoveryAuditRepository",
        "list_recoverable_runs",
        "complete_run",
        "append_run_event",
        "append_audit_event",
        "PLATFORM_RESTART_RECOVERY",
    )

    assert [phrase for phrase in required if phrase not in text] == []


def test_runtime_recovery_service_has_no_concrete_infrastructure_imports() -> None:
    text = RECOVERY_PATH.read_text(encoding="utf-8")
    forbidden = (
        "bot_platform_service.persistence",
        "bot_platform_service.infrastructure",
        "sqlalchemy",
        "asyncpg",
        "spot_grid_bot",
        "spot-greenwich-bot",
    )

    assert [phrase for phrase in forbidden if phrase in text] == []
