from __future__ import annotations

import asyncio
from pathlib import Path

from bot_platform_service.application import BotPlatformRunnerService
from bot_platform_service.config.settings import BotPlatformSettings
from bot_platform_service.domain import BotInstanceConfig, BotMode


REPO_ROOT = Path(__file__).resolve().parents[3]


class _InstanceRepository:
    def __init__(self, instances: tuple[BotInstanceConfig, ...]) -> None:
        self.instances = instances
        self.reads = 0

    async def list_enabled_instances(self) -> tuple[BotInstanceConfig, ...]:
        self.reads += 1
        return self.instances


class _MarketData:
    def __init__(self, *, ready: bool, status: str = "ready") -> None:
        self.ready = ready
        self.status = status
        self.readiness_checks = 0
        self.snapshot_checks: list[tuple[str, str, str]] = []

    async def is_ready(self) -> bool:
        self.readiness_checks += 1
        return self.ready

    async def latest_snapshot_status(self, *, source: str, canonical_symbol: str, timeframe: str) -> str:
        self.snapshot_checks.append((source, canonical_symbol, timeframe))
        return self.status


def test_stage_21_settings_and_cli_expose_runner(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PLATFORM_MARKET_DATA_BASE_URL", "http://market-data.local:8010/")
    monkeypatch.setenv("BOT_PLATFORM_MARKET_DATA_SOURCE", "BINANCE_SPOT")
    monkeypatch.setenv("BOT_PLATFORM_RUNNER_POLL_INTERVAL_SECONDS", "2.5")

    settings = BotPlatformSettings.from_env()
    cli_source = (REPO_ROOT / "bot-platform-service/src/bot_platform_service/cli.py").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "infra/compose/docker-compose.platform.yaml").read_text(encoding="utf-8")

    assert settings.market_data.base_url == "http://market-data.local:8010"
    assert settings.market_data.source == "BINANCE_SPOT"
    assert settings.runtime.runner_poll_interval_seconds == 2.5
    assert 'subparsers.add_parser("runner"' in cli_source
    assert "bot_platform_runner:" in compose
    assert 'command: ["python", "-m", "bot_platform_service.main", "runner"]' in compose


def test_stage_21_runner_checks_snapshot_availability_without_adapter_execution() -> None:
    async def run() -> None:
        instance = BotInstanceConfig(
            instance_id="instance-1",
            module_id="spot_grid",
            mode=BotMode.SIGNAL_ONLY,
            symbols=("ETHUSDT",),
            timeframes=("1h", "4h"),
            config_schema_version=1,
            config={},
        )
        market_data = _MarketData(ready=True, status="ready")
        service = BotPlatformRunnerService(
            instance_repository=_InstanceRepository((instance,)),
            market_data=market_data,
            source="BINANCE_SPOT",
        )

        result = await service.run_once()

        assert result.market_data_ready is True
        assert result.enabled_instance_count == 1
        assert [check.status for check in result.snapshot_checks] == ["ready", "ready"]
        assert market_data.snapshot_checks == [
            ("BINANCE_SPOT", "ETHUSDT", "1h"),
            ("BINANCE_SPOT", "ETHUSDT", "4h"),
        ]

    asyncio.run(run())


def test_stage_21_runner_skips_snapshots_when_market_data_unavailable() -> None:
    async def run() -> None:
        instance = BotInstanceConfig(
            instance_id="instance-1",
            module_id="spot_greenwich",
            mode=BotMode.NOTIFICATION_ONLY,
            symbols=("BTCUSDT",),
            timeframes=("1h",),
            config_schema_version=1,
            config={},
        )
        market_data = _MarketData(ready=False)
        service = BotPlatformRunnerService(
            instance_repository=_InstanceRepository((instance,)),
            market_data=market_data,
            source="BINANCE_SPOT",
        )

        result = await service.run_once()

        assert result.market_data_ready is False
        assert result.enabled_instance_count == 1
        assert result.snapshot_checks == ()
        assert market_data.snapshot_checks == []

    asyncio.run(run())
