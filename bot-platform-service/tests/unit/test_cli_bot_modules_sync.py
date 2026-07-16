from __future__ import annotations

import pytest

from bot_platform_service import cli
from bot_platform_service.main import main


def test_main_without_args_is_safe_noop() -> None:
    assert main([]) == 0


def test_bot_modules_sync_command_runs_discovery(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: list[str] = []

    async def fake_sync(database_url: str) -> tuple[str, ...]:
        calls.append(database_url)
        return ("spot_greenwich", "spot_grid")

    monkeypatch.setattr(cli, "get_database_url", lambda: "postgresql+asyncpg://example")
    monkeypatch.setattr(cli, "sync_bot_modules_from_database_url", fake_sync)

    exit_code = main(["bot:modules:sync"])

    assert exit_code == 0
    assert calls == ["postgresql+asyncpg://example"]
    assert "Synced bot modules: spot_greenwich, spot_grid" in capsys.readouterr().out
