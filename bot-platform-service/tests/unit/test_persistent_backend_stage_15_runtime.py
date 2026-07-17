from __future__ import annotations

import asyncio

from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.runtime import build_runtime_container


class _FakeConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeEngine:
    def __init__(self) -> None:
        self.connection = _FakeConnection()
        self.disposed = False

    async def connect(self) -> _FakeConnection:
        return self.connection

    async def dispose(self) -> None:
        self.disposed = True


def test_stage_15_settings_parse_typed_runtime_values(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PLATFORM_DATABASE_URL", "postgresql+asyncpg://user:pass@db:5432/app")
    monkeypatch.setenv("BOT_PLATFORM_HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("BOT_PLATFORM_HTTP_PORT", "18092")
    monkeypatch.setenv("BOT_PLATFORM_RUNNER_ENABLED", "false")

    settings = BotPlatformSettings.from_env()

    assert settings.database.url == "postgresql+asyncpg://user:pass@db:5432/app"
    assert settings.http.host == "127.0.0.1"
    assert settings.http.port == 18092
    assert settings.runtime.runner_enabled is False


def test_stage_15_composition_root_builds_and_closes_without_runner() -> None:
    fake_engine = _FakeEngine()
    settings = BotPlatformSettings.from_env()

    async def run() -> None:
        container = await build_runtime_container(settings, engine_factory=lambda _url: fake_engine)
        assert container.repositories.bot_modules.connection is fake_engine.connection
        assert container.admin_metadata_service.repository is container.repositories.bot_modules
        assert container.runner_started is False

        await container.close()
        assert fake_engine.connection.closed is True
        assert fake_engine.disposed is True

    asyncio.run(run())
