from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SMOKE_SCRIPT = REPO_ROOT / "infra" / "scripts" / "full-docker-smoke.sh"


def test_stage_26_full_docker_smoke_script_documents_required_flow() -> None:
    content = SMOKE_SCRIPT.read_text(encoding="utf-8")

    assert "market_data_migrate" in content
    assert "market_data_symbols_sync" in content
    assert "scheduler:run-once" in content
    assert "sync:run-next" in content
    assert "outbox:publish-once" in content
    assert "MARKET_DATA_PROVIDER_MODE" in content
    assert "bot_platform_migrate" in content
    assert "bot_modules_sync" in content
    assert "bot_platform_runner" in content
    assert "/admin/bot-modules" in content
    assert "/admin/bot-instances" in content
    assert "bot_runs" in content
    assert "bot_signals" in content
    assert "XLEN" in content
    assert "ADMIN_COOKIE_HEADER" in content
    assert "SMOKE_CLIENT_ORIGIN" in content
    assert "Origin: ${SMOKE_CLIENT_ORIGIN}" in content
    assert "item.instance_id || item.instanceId" in content
    assert "legacy bot services are running" in content


def test_stage_26_full_docker_smoke_script_is_executable() -> None:
    assert SMOKE_SCRIPT.exists()
    assert SMOKE_SCRIPT.stat().st_mode & 0o111
