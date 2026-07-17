from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.domain import BotInstanceConfig, BotInstanceStatus, BotMode


@dataclass(frozen=True, slots=True)
class AdminBotInstanceSummary:
    """Read-only admin summary for one configured bot instance."""

    instance_id: str
    module_id: str
    tenant_id: str | None
    name: str
    mode: BotMode
    status: BotInstanceStatus
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    config_schema_version: int
    config: Mapping[str, object]


class AdminBotInstanceRepository(Protocol):
    """Read boundary for admin bot instance list APIs."""

    async def list_instances(self) -> tuple[AdminBotInstanceSummary, ...]:
        """Return configured bot instances."""


class AdminBotInstanceService:
    """Read-only admin use cases for bot instances."""

    def __init__(self, *, repository: AdminBotInstanceRepository) -> None:
        self.repository = repository

    async def list_instances(self) -> tuple[AdminBotInstanceSummary, ...]:
        """Return configured bot instances without running bot code."""

        return await self.repository.list_instances()


def config_from_payload(payload: Mapping[str, object], *, instance_id: str) -> BotInstanceConfig:
    """Build a BotInstanceConfig from a validated admin payload."""

    return BotInstanceConfig(
        instance_id=instance_id,
        module_id=str(payload["module_id"]),
        mode=BotMode(str(payload["mode"])),
        symbols=tuple(str(symbol).upper() for symbol in _string_list(payload["symbols"])),
        timeframes=tuple(str(timeframe) for timeframe in _string_list(payload["timeframes"])),
        config_schema_version=int(payload["config_schema_version"]),
        config=dict(payload["config"]) if isinstance(payload["config"], Mapping) else {},
        tenant_id=str(payload["tenant_id"]) if payload.get("tenant_id") else None,
        name=str(payload["name"]) if payload.get("name") else None,
    )


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())
