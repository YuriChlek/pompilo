from __future__ import annotations

from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[2]
ORCHESTRATION_PATH = SERVICE_ROOT / "src" / "bot_platform_service" / "application" / "bot_run_orchestration_service.py"


def test_scheduler_event_routing_service_exists() -> None:
    assert ORCHESTRATION_PATH.exists()


def test_scheduler_event_routing_stays_in_application_layer() -> None:
    text = ORCHESTRATION_PATH.read_text(encoding="utf-8")
    forbidden = (
        "bot_platform_service.infrastructure.bot_modules",
        "spot_grid_bot",
        "spot-greenwich-bot",
        "spot_greenwich",
        "sqlalchemy",
        "asyncpg",
    )

    assert [phrase for phrase in forbidden if phrase in text] == []


def test_scheduler_event_routing_exports_required_contracts() -> None:
    text = ORCHESTRATION_PATH.read_text(encoding="utf-8")
    required = (
        "CandleBatchReadyEvent",
        "PollingScheduleTick",
        "BotRunOrchestrationService",
        "build_trigger_idempotency_key",
        "find_stuck_runs",
        "list_enabled_instances_for_snapshot",
    )

    assert [phrase for phrase in required if phrase not in text] == []
