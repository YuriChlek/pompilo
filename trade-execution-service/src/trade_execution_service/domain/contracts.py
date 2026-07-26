from __future__ import annotations

from typing import Protocol

from trade_execution_service.domain.models import (
    BalanceSnapshot,
    ExecutionAccountRef,
    ExecutionDecision,
    ExecutionIntent,
    PersistedBotSignalRef,
    PositionSnapshot,
    VenueConstraints,
)


class SignalPayloadStore(Protocol):
    """Reads full persisted signal payloads from Bot Platform storage."""

    async def load_signal_payload(self, *, signal_id: str) -> tuple[PersistedBotSignalRef, dict[str, object]]:
        """Return signal metadata and full JSON payload."""


class ExecutionAccountResolver(Protocol):
    """Resolves the exchange account permitted for a bot signal."""

    async def resolve_account(self, *, signal: PersistedBotSignalRef) -> ExecutionAccountRef:
        """Return the execution account for one signal."""


class ExchangeExecutionAdapter(Protocol):
    """Exchange-specific private execution adapter."""

    async def get_balances(self, *, account: ExecutionAccountRef) -> tuple[BalanceSnapshot, ...]:
        """Return normalized account balances."""

    async def get_position(self, *, account: ExecutionAccountRef, symbol: str) -> PositionSnapshot:
        """Return normalized position state for one symbol."""

    async def get_constraints(self, *, account: ExecutionAccountRef, symbol: str) -> VenueConstraints:
        """Return normalized venue constraints for one symbol."""

    async def execute(self, *, account: ExecutionAccountRef, intent: ExecutionIntent) -> ExecutionDecision:
        """Execute an already validated intent on the exchange."""


class ExecutionAuditLog(Protocol):
    """Persists execution decisions and venue outcomes."""

    async def append_decision(self, *, decision: ExecutionDecision) -> None:
        """Record one execution decision."""


class ExecutionControlPolicy(Protocol):
    """Reads runtime execution kill switches and pause controls."""

    async def is_global_kill_switch_enabled(self) -> bool:
        """Return true when all execution must be disabled."""

    async def is_bot_or_symbol_paused(self, *, signal: PersistedBotSignalRef) -> bool:
        """Return true when this bot instance or symbol is paused for execution."""


class ExecutionIdempotencyStore(Protocol):
    """Tracks processed persisted bot signals before venue execution."""

    async def was_processed(self, *, signal_id: str, signal_key: str | None) -> bool:
        """Return true when this signal was already handled by execution service."""

    async def mark_processed(self, *, signal_id: str, signal_key: str | None) -> None:
        """Mark this signal as handled."""


class ExecutionStatusPublisher(Protocol):
    """Publishes execution status transitions for downstream audit/ops views."""

    async def publish_status(self, *, decision: ExecutionDecision) -> None:
        """Publish one accepted/rejected/placed/filled/cancelled/failed/skipped status."""
