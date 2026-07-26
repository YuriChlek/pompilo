from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from bot_platform_service.domain import BotHealthStatus, BotInstanceStatus, BotModuleStatus, BotRunStatus


class OperationalStatusProvider(Protocol):
    """Read model provider for platform health state."""

    async def failed_modules(self) -> tuple["OperationalIssue", ...]:
        """Return failed or disabled module issues."""

    async def failed_instances(self) -> tuple["OperationalIssue", ...]:
        """Return failed instance issues."""

    async def failed_runs(self) -> tuple["OperationalIssue", ...]:
        """Return failed run issues."""


@dataclass(frozen=True, slots=True)
class OperationalIssue:
    """Operator-visible failed entity and reason."""

    entity_type: str
    entity_id: str
    status: str
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlatformHealthReport:
    """Aggregated health response for the platform health endpoint."""

    status: BotHealthStatus
    failed_modules: tuple[OperationalIssue, ...] = ()
    failed_instances: tuple[OperationalIssue, ...] = ()
    failed_runs: tuple[OperationalIssue, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready health payload."""

        return {
            "status": self.status.value,
            "failed_modules": [_issue_to_dict(issue) for issue in self.failed_modules],
            "failed_instances": [_issue_to_dict(issue) for issue in self.failed_instances],
            "failed_runs": [_issue_to_dict(issue) for issue in self.failed_runs],
        }


class PlatformHealthService:
    """Build operator-visible health reports from read-model providers."""

    def __init__(self, provider: OperationalStatusProvider) -> None:
        self.provider = provider

    async def report(self) -> PlatformHealthReport:
        """Return aggregated platform health."""

        failed_modules = await self.provider.failed_modules()
        failed_instances = await self.provider.failed_instances()
        failed_runs = await self.provider.failed_runs()
        status = BotHealthStatus.HEALTHY
        if failed_modules or failed_instances or failed_runs:
            status = BotHealthStatus.UNHEALTHY
        return PlatformHealthReport(
            status=status,
            failed_modules=failed_modules,
            failed_instances=failed_instances,
            failed_runs=failed_runs,
        )


class HealthEndpoint:
    """Framework-neutral health endpoint adapter."""

    def __init__(self, health_service: PlatformHealthService) -> None:
        self.health_service = health_service

    async def get(self) -> tuple[int, dict[str, object]]:
        """Return HTTP-like status code and JSON body."""

        report = await self.health_service.report()
        status_code = 200 if report.status is BotHealthStatus.HEALTHY else 503
        return status_code, report.to_dict()


def module_issue(module_id: str, status: BotModuleStatus, reason: str) -> OperationalIssue:
    return OperationalIssue("module", module_id, status.value, reason)


def instance_issue(instance_id: str, status: BotInstanceStatus, reason: str) -> OperationalIssue:
    return OperationalIssue("instance", instance_id, status.value, reason)


def run_issue(run_id: str, status: BotRunStatus, reason: str) -> OperationalIssue:
    return OperationalIssue("run", run_id, status.value, reason)


def _issue_to_dict(issue: OperationalIssue) -> dict[str, object]:
    return {
        "entity_type": issue.entity_type,
        "entity_id": issue.entity_id,
        "status": issue.status,
        "reason": issue.reason,
        "details": dict(issue.details),
    }


__all__ = [
    "HealthEndpoint",
    "OperationalIssue",
    "OperationalStatusProvider",
    "PlatformHealthReport",
    "PlatformHealthService",
    "instance_issue",
    "module_issue",
    "run_issue",
]
