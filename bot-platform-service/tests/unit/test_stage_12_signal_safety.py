from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from bot_platform_service.domain import BotMode, BotSignal, BotSignalSide, BotSignalType, resolve_bot_mode
from bot_platform_service.infrastructure.signals import PersistentSignalPublisher, build_signal_id


class _SignalRepository:
    def __init__(self) -> None:
        self.signal_ids_by_key: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []

    async def publish_signal(self, *, signal_id, run_id, signal, status, correlation_id=None):
        self.calls.append((signal.signal_key, signal_id))
        return self.signal_ids_by_key.setdefault(signal.signal_key, signal_id)


class _AuditRepository:
    def __init__(self) -> None:
        self.events_by_id: dict[str, dict[str, object]] = {}
        self.calls: list[str] = []

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
        self.calls.append(event_id)
        if event_id in self.events_by_id:
            return False
        self.events_by_id[event_id] = {
            "event_type": event_type,
            "payload_json": payload_json,
            "instance_id": instance_id,
            "module_id": module_id,
        }
        return True


def test_signal_safe_mode_defaults_to_dry_run() -> None:
    assert resolve_bot_mode(None) is BotMode.DRY_RUN
    assert resolve_bot_mode("") is BotMode.DRY_RUN
    assert resolve_bot_mode("signal_only") is BotMode.SIGNAL_ONLY


def test_signal_id_is_deterministic_from_signal_key() -> None:
    signal = _signal()

    assert build_signal_id(signal) == build_signal_id(signal)
    assert build_signal_id(signal).startswith("sig_")


def test_persistent_signal_publisher_is_idempotent_and_audited() -> None:
    signal_repository = _SignalRepository()
    audit_repository = _AuditRepository()
    publisher = PersistentSignalPublisher(
        signal_repository=signal_repository,
        audit_repository=audit_repository,
        run_id="run-1",
        correlation_id="corr-1",
    )
    signal = _signal()

    first = asyncio.run(publisher.publish(signal))
    second = asyncio.run(publisher.publish(signal))

    assert first.accepted is True
    assert second.accepted is True
    assert first.signal_id == second.signal_id
    assert len(signal_repository.calls) == 2
    assert len(audit_repository.events_by_id) == 1
    event = next(iter(audit_repository.events_by_id.values()))
    assert event["event_type"] == "SIGNAL_PERSISTED"
    assert event["instance_id"] == signal.instance_id
    assert event["module_id"] == signal.module_id
    assert event["payload_json"]["signal_key"] == signal.signal_key
    assert event["payload_json"]["boundary"] == "execution_service_reads_signals_only"


def test_persistent_signal_publisher_preserves_payload_schema_version() -> None:
    audit_repository = _AuditRepository()
    publisher = PersistentSignalPublisher(
        signal_repository=_SignalRepository(),
        audit_repository=audit_repository,
        run_id="run-1",
    )
    signal = _signal(payload_schema_version=7)

    asyncio.run(publisher.publish(signal))

    event = next(iter(audit_repository.events_by_id.values()))
    assert event["payload_json"]["payload_schema"] == "stage12.test"
    assert event["payload_json"]["payload_schema_version"] == 7
    assert event["payload_json"]["payload_hash"] == signal.payload_hash


def _signal(*, payload_schema_version: int = 1) -> BotSignal:
    return BotSignal.build(
        instance_id="instance-1",
        module_id="spot_grid_bot",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.8"),
        reason="stage12",
        payload_schema="stage12.test",
        payload_schema_version=payload_schema_version,
        payload={
            "price": Decimal("100.5"),
            "generated_at": datetime(2026, 7, 14, tzinfo=UTC),
        },
    )
