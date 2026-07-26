from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    """Readiness result for the Bot Platform HTTP API."""

    ready: bool
    checks: dict[str, bool]
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready readiness payload."""

        payload: dict[str, object] = {
            "status": "ready" if self.ready else "not_ready",
            "checks": dict(self.checks),
        }
        if self.errors:
            payload["errors"] = dict(self.errors)
        return payload


class PostgresReadinessProbe:
    """Readiness probe backed by a PostgreSQL `select 1` check."""

    def __init__(self, connection: AsyncConnection) -> None:
        self.connection = connection

    async def check(self) -> ReadinessReport:
        """Check whether PostgreSQL can serve a trivial query."""

        try:
            await self.connection.execute(text("select 1"))
        except Exception as exc:  # noqa: BLE001 - readiness must report the concrete startup failure.
            return ReadinessReport(
                ready=False,
                checks={"postgres": False},
                errors={"postgres": exc.__class__.__name__},
            )
        return ReadinessReport(ready=True, checks={"postgres": True})
