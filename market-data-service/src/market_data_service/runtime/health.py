from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from market_data_service.runtime.container import MarketDataRuntimeContainer
from market_data_service.runtime.lifecycle import RuntimeLifecycle

EXPECTED_ALEMBIC_REVISION = "0010_normalize_symbol_format"


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    ok: bool
    message: str


@dataclass(frozen=True, slots=True)
class HealthStatus:
    ok: bool
    checks: tuple[HealthCheck, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "status": "ready" if self.ok else "not_ready",
            "checks": [
                {
                    "name": check.name,
                    "ok": check.ok,
                    "message": check.message,
                }
                for check in self.checks
            ],
        }


class RuntimeHealthService:
    def __init__(
        self,
        container: MarketDataRuntimeContainer,
        lifecycle: RuntimeLifecycle | None = None,
        *,
        scheduler_enabled: bool = False,
        outbox_publisher_enabled: bool = False,
    ) -> None:
        self.container = container
        self.lifecycle = lifecycle
        self.scheduler_enabled = scheduler_enabled
        self.outbox_publisher_enabled = outbox_publisher_enabled

    async def readiness(self) -> HealthStatus:
        if self.lifecycle is not None and self.lifecycle.is_shutting_down:
            return HealthStatus(
                ok=False,
                checks=(HealthCheck(name="process_state", ok=False, message=self.lifecycle.state.value),),
            )
        checks = [
            await self._check_database(),
            await self._check_redis(),
            await self._check_migration_revision(),
        ]
        if self.scheduler_enabled:
            checks.append(self._check_scheduler())
        if self.outbox_publisher_enabled:
            checks.append(self._check_outbox_publisher())
        return HealthStatus(ok=all(check.ok for check in checks), checks=checks)

    async def _check_database(self) -> HealthCheck:
        try:
            await self.container.connection.execute(text("select 1"))
        except Exception as exc:
            return HealthCheck(name="postgres", ok=False, message=f"{type(exc).__name__}: {exc}")
        return HealthCheck(name="postgres", ok=True, message="ok")

    async def _check_redis(self) -> HealthCheck:
        try:
            pong = await _maybe_await(self.container.redis_client.ping())
        except Exception as exc:
            return HealthCheck(name="redis", ok=False, message=f"{type(exc).__name__}: {exc}")
        if pong not in (True, "PONG", b"PONG"):
            return HealthCheck(name="redis", ok=False, message=f"unexpected ping response: {pong!r}")
        return HealthCheck(name="redis", ok=True, message="ok")

    async def _check_migration_revision(self) -> HealthCheck:
        try:
            result = await self.container.connection.execute(
                text("select version_num from market_data_alembic_version")
            )
            revisions = {row[0] for row in result}
        except Exception as exc:
            return HealthCheck(name="migration_revision", ok=False, message=f"{type(exc).__name__}: {exc}")
        if EXPECTED_ALEMBIC_REVISION not in revisions:
            return HealthCheck(
                name="migration_revision",
                ok=False,
                message=f"expected {EXPECTED_ALEMBIC_REVISION}, got {', '.join(sorted(revisions)) or '<none>'}",
            )
        return HealthCheck(name="migration_revision", ok=True, message=EXPECTED_ALEMBIC_REVISION)

    def _check_scheduler(self) -> HealthCheck:
        scheduler_lifecycle = getattr(self.container, "scheduler_lifecycle", None)
        if scheduler_lifecycle is None:
            return HealthCheck(name="scheduler", ok=False, message="scheduler lifecycle is not configured")
        health = scheduler_lifecycle.health()
        if not health.started:
            return HealthCheck(name="scheduler", ok=False, message="not_started")
        if health.last_tick_ok is False:
            return HealthCheck(name="scheduler", ok=True, message=f"degraded: {health.last_error}")
        return HealthCheck(name="scheduler", ok=True, message="ok")

    def _check_outbox_publisher(self) -> HealthCheck:
        outbox_lifecycle = getattr(self.container, "outbox_publisher_lifecycle", None)
        if outbox_lifecycle is None:
            return HealthCheck(name="outbox_publisher", ok=False, message="outbox publisher lifecycle is not configured")
        health = outbox_lifecycle.health()
        if not health.started:
            return HealthCheck(name="outbox_publisher", ok=False, message="not_started")
        if health.last_publish_ok is False:
            return HealthCheck(name="outbox_publisher", ok=True, message=f"degraded: {health.last_error}")
        return HealthCheck(name="outbox_publisher", ok=True, message="ok")


async def _maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
