from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bot_platform_service.domain import BotInstanceStatus, BotMode
from bot_platform_service.persistence.repositories import BotInstanceRepository


class _FakeResult:
    def __init__(self, *, rowcount: int = 1, row=None) -> None:
        self.rowcount = rowcount
        self._row = row

    def first(self):
        return self._row


class _FakeConnection:
    def __init__(self, results: list[_FakeResult]) -> None:
        self.results = list(results)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return self.results.pop(0)


def test_instance_repository_reads_active_config_and_status() -> None:
    connection = _FakeConnection(
        [
            _FakeResult(row=SimpleNamespace(status="ENABLED")),
            _FakeResult(
                row=SimpleNamespace(
                    instance_id="instance-1",
                    module_id="spot_grid_bot",
                    tenant_id=None,
                    name="Grid",
                    mode="dry_run",
                    symbols=["ETHUSDT"],
                    timeframes=["1h", "4h"],
                    config_schema_version=1,
                    config_json={"risk": "low"},
                )
            ),
        ]
    )
    repository = BotInstanceRepository(connection)

    status = asyncio.run(repository.get_instance_status("instance-1"))
    config = asyncio.run(repository.get_instance_config("instance-1"))

    assert status is BotInstanceStatus.ENABLED
    assert config is not None
    assert config.mode is BotMode.DRY_RUN
    assert config.symbols == ("ETHUSDT",)
    assert config.config == {"risk": "low"}
    assert len(connection.statements) == 2


def test_instance_repository_updates_status() -> None:
    connection = _FakeConnection([_FakeResult(rowcount=1)])
    repository = BotInstanceRepository(connection)

    changed = asyncio.run(repository.update_instance_status(instance_id="instance-1", status=BotInstanceStatus.PAUSED))

    assert changed is True
    assert len(connection.statements) == 1
