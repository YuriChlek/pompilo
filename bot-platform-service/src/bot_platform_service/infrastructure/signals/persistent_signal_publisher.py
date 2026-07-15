from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.domain import BotSignal, BotSignalPublishResult, build_payload_hash
from bot_platform_service.domain.enums import BotSignalPublishStatus


class SignalRepository(Protocol):
    """Repository boundary for idempotent signal persistence."""

    async def publish_signal(
        self,
        *,
        signal_id: str,
        run_id: str,
        signal: BotSignal,
        status: BotSignalPublishStatus = BotSignalPublishStatus.PUBLISHED,
        correlation_id: str | None = None,
    ) -> str:
        """Persist one signal idempotently and return its stable signal id."""


class SignalAuditRepository(Protocol):
    """Repository boundary for append-only signal audit events."""

    async def append_audit_event(
        self,
        *,
        event_id: str,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload_json: dict[str, object],
        instance_id: str | None = None,
        module_id: str | None = None,
        correlation_id: str | None = None,
    ) -> bool:
        """Append one duplicate-safe audit event."""


@dataclass(frozen=True, slots=True)
class PersistentSignalPublisher:
    """Run-scoped publisher that persists signals and writes audit trail."""

    signal_repository: SignalRepository
    audit_repository: SignalAuditRepository
    run_id: str
    actor_type: str = "bot_platform"
    actor_id: str = "signal_publisher"
    correlation_id: str | None = None

    async def publish(self, signal: BotSignal) -> BotSignalPublishResult:
        """Persist one normalized signal idempotently and audit the persisted signal."""

        signal_id = build_signal_id(signal)
        persisted_signal_id = await self.signal_repository.publish_signal(
            signal_id=signal_id,
            run_id=self.run_id,
            signal=signal,
            status=BotSignalPublishStatus.PUBLISHED,
            correlation_id=self.correlation_id,
        )
        await self.audit_repository.append_audit_event(
            event_id=f"signal_persisted:{persisted_signal_id}",
            event_type="SIGNAL_PERSISTED",
            actor_type=self.actor_type,
            actor_id=self.actor_id,
            instance_id=signal.instance_id,
            module_id=signal.module_id,
            payload_json={
                "signal_id": persisted_signal_id,
                "signal_key": signal.signal_key,
                "run_id": self.run_id,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "snapshot_id": signal.snapshot_id,
                "signal_type": signal.signal_type.value,
                "side": signal.side.value if signal.side is not None else None,
                "payload_schema": signal.payload_schema,
                "payload_schema_version": signal.payload_schema_version,
                "payload_hash": signal.payload_hash,
                "boundary": "execution_service_reads_signals_only",
            },
            correlation_id=self.correlation_id,
        )
        return BotSignalPublishResult(accepted=True, signal_id=persisted_signal_id)


def build_signal_id(signal: BotSignal) -> str:
    """Build a stable signal id from the deterministic signal key."""

    return f"sig_{build_payload_hash({'signal_key': signal.signal_key})[:32]}"
