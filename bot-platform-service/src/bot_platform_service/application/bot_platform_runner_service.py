from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.domain import BotInstanceConfig


class RunnerInstanceRepository(Protocol):
    """Read boundary for runner instance discovery."""

    async def list_enabled_instances(self) -> tuple[BotInstanceConfig, ...]:
        """Return all enabled instances with active configs."""


class MarketDataSnapshotReadinessPort(Protocol):
    """Boundary for Market Data Service readiness and snapshot availability checks."""

    async def is_ready(self) -> bool:
        """Return whether Market Data Service is ready for snapshot reads."""

    async def latest_snapshot_status(self, *, source: str, canonical_symbol: str, timeframe: str) -> str:
        """Return latest snapshot contract status for one symbol/timeframe."""


@dataclass(frozen=True, slots=True)
class RunnerSnapshotCheck:
    """One runner snapshot availability check."""

    instance_id: str
    module_id: str
    canonical_symbol: str
    timeframe: str
    status: str


@dataclass(frozen=True, slots=True)
class RunnerTickResult:
    """Summary of one runner scheduler tick."""

    market_data_ready: bool
    enabled_instance_count: int
    snapshot_checks: tuple[RunnerSnapshotCheck, ...]


class BotPlatformRunnerService:
    """Runner skeleton that checks enabled instances without executing bot adapters."""

    def __init__(
        self,
        *,
        instance_repository: RunnerInstanceRepository,
        market_data: MarketDataSnapshotReadinessPort,
        source: str,
    ) -> None:
        self.instance_repository = instance_repository
        self.market_data = market_data
        self.source = source

    async def run_once(self) -> RunnerTickResult:
        """Check market-data availability for enabled instances and return a summary."""

        instances = await self.instance_repository.list_enabled_instances()
        try:
            market_data_ready = await self.market_data.is_ready()
        except Exception:
            return RunnerTickResult(
                market_data_ready=False,
                enabled_instance_count=len(instances),
                snapshot_checks=(),
            )
        if not market_data_ready:
            return RunnerTickResult(
                market_data_ready=False,
                enabled_instance_count=len(instances),
                snapshot_checks=(),
            )

        checks: list[RunnerSnapshotCheck] = []
        for instance in instances:
            for canonical_symbol in instance.symbols:
                for timeframe in instance.timeframes:
                    checks.append(
                        await self._check_snapshot(
                            instance=instance,
                            canonical_symbol=canonical_symbol,
                            timeframe=timeframe,
                        )
                    )
        return RunnerTickResult(
            market_data_ready=True,
            enabled_instance_count=len(instances),
            snapshot_checks=tuple(checks),
        )

    async def _check_snapshot(
        self,
        *,
        instance: BotInstanceConfig,
        canonical_symbol: str,
        timeframe: str,
    ) -> RunnerSnapshotCheck:
        try:
            status = await self.market_data.latest_snapshot_status(
                source=self.source,
                canonical_symbol=canonical_symbol,
                timeframe=timeframe,
            )
        except Exception:
            status = "unavailable"
        return RunnerSnapshotCheck(
            instance_id=instance.instance_id,
            module_id=instance.module_id,
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            status=status,
        )
