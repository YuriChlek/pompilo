from __future__ import annotations

import asyncio
from decimal import Decimal

from bot_platform_service.domain import BotSignal, BotSignalSide, BotSignalType
from bot_platform_service.infrastructure.signals import RedisStreamSignalEventPublisher
from bot_platform_service.persistence.repositories import BotSignalRepository
from bot_platform_service.trading_bots.spot_grid.domain import (
    POSITION_INTENT_ENTRY_EXAMPLE,
    POSITION_INTENT_PAYLOAD_SCHEMA,
    POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
)


class _Mappings:
    def __init__(self, row: dict[str, object] | None) -> None:
        self.row = row

    def first(self) -> dict[str, object] | None:
        return self.row


class _Result:
    def __init__(
        self,
        *,
        rowcount: int = 1,
        scalar_value: object | None = None,
        row: dict[str, object] | None = None,
    ) -> None:
        self.rowcount = rowcount
        self.scalar_value = scalar_value
        self.row = row

    def scalar_one(self) -> object:
        if self.scalar_value is None:
            raise AssertionError("No scalar value configured")
        return self.scalar_value

    def mappings(self) -> _Mappings:
        return _Mappings(self.row)


class _Connection:
    def __init__(self, *, persisted_row: dict[str, object]) -> None:
        self.persisted_row = persisted_row
        self.execute_count = 0
        self.statements: list[object] = []

    async def execute(self, statement):
        self.execute_count += 1
        self.statements.append(statement)
        if self.execute_count == 1:
            return _Result(rowcount=1)
        if self.execute_count == 2:
            return _Result(rowcount=0)
        if self.execute_count == 3:
            return _Result(rowcount=0, scalar_value=self.persisted_row["signal_id"])
        return _Result(row=self.persisted_row)


class _Redis:
    def __init__(self) -> None:
        self.keys: set[str] = set()
        self.events: list[tuple[str, dict[str, str]]] = []

    async def set(self, key: str, value: str, *, nx: bool = False):
        if nx and key in self.keys:
            return False
        self.keys.add(key)
        return True

    async def xadd(self, stream_name: str, fields: dict[str, str]):
        self.events.append((stream_name, dict(fields)))
        return "1-0"

    async def delete(self, key: str):
        self.keys.discard(key)
        return 1


def test_phase_29_persisted_signal_event_contract_and_payload_lookup() -> None:
    async def run() -> None:
        signal = _signal()
        persisted_signal_id = "sig-phase-29"
        row = _persisted_row(signal=signal, signal_id=persisted_signal_id)
        signal_repository = BotSignalRepository(_Connection(persisted_row=row))
        redis = _Redis()
        event_publisher = RedisStreamSignalEventPublisher(
            redis_client=redis,
            stream_name="bot-platform-signals",
            max_retries=0,
            retry_backoff_seconds=0,
        )

        first_signal_id = await signal_repository.publish_signal(
            signal_id=persisted_signal_id,
            run_id="run-phase-29-a",
            signal=signal,
        )
        second_signal_id = await signal_repository.publish_signal(
            signal_id="different-candidate-id",
            run_id="run-phase-29-b",
            signal=signal,
        )
        first_event_published = await event_publisher.publish_persisted_signal(
            signal_id=first_signal_id,
            run_id="run-phase-29-a",
            signal=signal,
            correlation_id="corr-phase-29",
        )
        second_event_published = await event_publisher.publish_persisted_signal(
            signal_id=second_signal_id,
            run_id="run-phase-29-b",
            signal=signal,
            correlation_id="corr-phase-29",
        )
        persisted = await signal_repository.get_signal_by_id(first_signal_id)

        assert first_signal_id == persisted_signal_id
        assert second_signal_id == persisted_signal_id
        assert first_event_published is True
        assert second_event_published is False
        assert len(redis.events) == 1

        event = redis.events[0][1]
        assert event == {
            "event_type": "bot_signal.persisted.v1",
            "idempotency_key": "bot-platform:signal-event:sig-phase-29",
            "signal_id": persisted_signal_id,
            "signal_key": signal.signal_key,
            "run_id": "run-phase-29-a",
            "instance_id": "instance-1",
            "module_id": "spot_grid",
            "symbol": "ETHUSDT",
            "timeframe": "1h",
            "snapshot_id": "snapshot-phase-29",
            "signal_type": "entry",
            "side": "buy",
            "payload_schema": POSITION_INTENT_PAYLOAD_SCHEMA,
            "payload_schema_version": str(POSITION_INTENT_PAYLOAD_SCHEMA_VERSION),
            "payload_hash": signal.payload_hash,
            "correlation_id": "corr-phase-29",
        }
        assert "payload" not in event
        assert "payload_json" not in event
        assert "target_price" not in event

        assert persisted is not None
        assert persisted["signal_id"] == persisted_signal_id
        assert persisted["payload_schema"] == POSITION_INTENT_PAYLOAD_SCHEMA
        assert persisted["payload_schema_version"] == POSITION_INTENT_PAYLOAD_SCHEMA_VERSION
        assert persisted["payload_hash"] == signal.payload_hash
        assert persisted["payload_json"] == dict(signal.payload)
        assert persisted["payload_json"]["target_price"] == "3180.50"

    asyncio.run(run())


def _signal() -> BotSignal:
    return BotSignal.build(
        instance_id="instance-1",
        module_id="spot_grid",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-phase-29",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.80"),
        reason="range_buy",
        payload_schema=POSITION_INTENT_PAYLOAD_SCHEMA,
        payload_schema_version=POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
        payload=dict(POSITION_INTENT_ENTRY_EXAMPLE),
    )


def _persisted_row(*, signal: BotSignal, signal_id: str) -> dict[str, object]:
    return {
        "signal_id": signal_id,
        "signal_key": signal.signal_key,
        "run_id": "run-phase-29-a",
        "instance_id": signal.instance_id,
        "module_id": signal.module_id,
        "symbol": signal.symbol,
        "timeframe": signal.timeframe,
        "snapshot_id": signal.snapshot_id,
        "signal_type": signal.signal_type.value,
        "side": signal.side.value if signal.side is not None else None,
        "confidence": signal.confidence,
        "reason": signal.reason,
        "payload_schema": signal.payload_schema,
        "payload_schema_version": signal.payload_schema_version,
        "payload_hash": signal.payload_hash,
        "payload_json": dict(signal.payload),
        "status": "PUBLISHED",
        "correlation_id": "corr-phase-29",
    }
