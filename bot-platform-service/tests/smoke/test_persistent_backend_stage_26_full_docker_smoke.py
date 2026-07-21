from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_SCRIPT = REPO_ROOT / "infra" / "scripts" / "full-docker-smoke.sh"
APPS_COMPOSE = REPO_ROOT / "infra" / "compose" / "docker-compose.apps.yaml"


def test_stage_26_full_docker_smoke_script_documents_required_flow() -> None:
    content = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "market_data_migrate" in content
    assert "collect --once" in content
    assert "MARKET_DATA_PROVIDER_MODE" in content
    assert "SMOKE_CANONICAL_SYMBOL" in content
    assert "reset_market_data_smoke_state" in content
    assert "seed_market_data_smoke_symbol" in content
    assert "reset_bot_platform_smoke_state" in content
    assert "bootstrap Redis stream" in content
    assert "bot_runs before incremental market-data event" in content
    assert "prepare_incremental_gap" in content
    assert "bot_platform_migrate" in content
    assert "bot_modules_sync" in content
    assert "bot_platform_runner" in content
    assert "/admin/bot-modules" in content
    assert "/admin/bot-instances" in content
    assert "bot_runs" in content
    assert "bot_signals" in content
    assert "trigger_type = 'event'" in content
    assert "market_data_processed_events" in content
    assert "events:cleanup" in content
    assert "event bot_runs after idempotent rerun" in content
    assert "XLEN" in content
    assert "ADMIN_COOKIE_HEADER" in content
    assert "SMOKE_CLIENT_ORIGIN" in content
    assert "Origin: process.env.SMOKE_CLIENT_ORIGIN" in content
    assert "item.instance_id || item.instanceId" in content
    assert "legacy bot services are running" in content
    assert "Full Docker event-driven smoke passed." in content
    assert "/run\"" not in content
    assert "symbols:sync" not in content
    assert "scheduler:run-once" not in content
    assert "sync:run-next" not in content
    assert "outbox:publish-once" not in content


def test_stage_26_full_docker_smoke_script_is_executable() -> None:
    assert SMOKE_SCRIPT.exists()
    assert SMOKE_SCRIPT.stat().st_mode & 0o111


def test_identity_api_compose_provides_required_auth_environment() -> None:
    content = APPS_COMPOSE.read_text(encoding="utf-8")

    assert "COOKIE_DOMAIN:" in content
    assert "JWT_SECRET:" in content
    assert "JWT_ACCESS_TOKEN_TTL:" in content
    assert "JWT_REFRESH_TOKEN_TTL:" in content
    assert "SESSION_MAX_TTL:" in content
    assert "DEVICE_ID_COOKIE_TTL:" in content
    assert "ENCRYPTION_KEY:" in content
