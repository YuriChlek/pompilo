from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotSignal, BotSignalSide, BotSignalType, canonical_json
from bot_platform_service.infrastructure.signals import RedisStreamSignalEventPublisher
from bot_platform_service.trading_bots.spot_grid.domain import (
    POSITION_INTENT_ENTRY_EXAMPLE,
    POSITION_INTENT_EXIT_EXAMPLE,
    POSITION_INTENT_PAYLOAD_SCHEMA,
    POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DOC = SERVICE_ROOT / "docs" / "spot_grid_position_intent_contract.md"


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


def test_phase_5_spot_grid_position_intent_schema_identity_is_versioned() -> None:
    assert POSITION_INTENT_PAYLOAD_SCHEMA == "spot_grid.position_intent"
    assert POSITION_INTENT_PAYLOAD_SCHEMA_VERSION == 1

    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    assert "`payload_schema`: `spot_grid.position_intent`" in doc
    assert "`payload_schema_version`: `1`" in doc
    assert "Any semantic payload shape\nchange must increment `payload_schema_version`" in doc


def test_phase_5_spot_grid_position_intent_examples_match_bot_signal_constraints() -> None:
    entry = _build_signal(
        payload=dict(POSITION_INTENT_ENTRY_EXAMPLE),
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
    )
    exit_signal = _build_signal(
        payload=dict(POSITION_INTENT_EXIT_EXAMPLE),
        signal_type=BotSignalType.EXIT,
        side=BotSignalSide.SELL,
    )

    assert entry.payload_schema == POSITION_INTENT_PAYLOAD_SCHEMA
    assert entry.payload_schema_version == POSITION_INTENT_PAYLOAD_SCHEMA_VERSION
    assert entry.payload_hash == _build_signal(
        payload=dict(POSITION_INTENT_ENTRY_EXAMPLE),
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
    ).payload_hash
    assert exit_signal.payload_schema == POSITION_INTENT_PAYLOAD_SCHEMA
    assert exit_signal.payload_schema_version == POSITION_INTENT_PAYLOAD_SCHEMA_VERSION
    assert canonical_json(POSITION_INTENT_ENTRY_EXAMPLE)
    assert canonical_json(POSITION_INTENT_EXIT_EXAMPLE)


def test_phase_5_spot_grid_position_intent_examples_have_entry_and_exit_semantics() -> None:
    assert POSITION_INTENT_ENTRY_EXAMPLE["intent_type"] == "open_position"
    assert POSITION_INTENT_ENTRY_EXAMPLE["side"] == "buy"
    assert POSITION_INTENT_ENTRY_EXAMPLE["symbol"] == "ETHUSDT"
    assert POSITION_INTENT_EXIT_EXAMPLE["intent_type"] == "close_position"
    assert POSITION_INTENT_EXIT_EXAMPLE["side"] == "sell"
    assert POSITION_INTENT_EXIT_EXAMPLE["symbol"] == "ETHUSDT"


def test_phase_5_persisted_signal_event_contains_metadata_not_full_payload() -> None:
    async def run() -> None:
        redis = _Redis()
        publisher = RedisStreamSignalEventPublisher(
            redis_client=redis,
            stream_name="bot-platform-signals",
            max_retries=0,
            retry_backoff_seconds=0,
        )
        signal = _build_signal(
            payload=dict(POSITION_INTENT_ENTRY_EXAMPLE),
            signal_type=BotSignalType.ENTRY,
            side=BotSignalSide.BUY,
        )

        accepted = await publisher.publish_persisted_signal(signal_id="sig-phase-5", run_id="run-1", signal=signal)

        assert accepted is True
        event = redis.events[0][1]
        assert event["event_type"] == "bot_signal.persisted.v1"
        assert event["signal_id"] == "sig-phase-5"
        assert event["payload_schema"] == POSITION_INTENT_PAYLOAD_SCHEMA
        assert event["payload_schema_version"] == str(POSITION_INTENT_PAYLOAD_SCHEMA_VERSION)
        assert event["payload_hash"] == signal.payload_hash
        assert "payload" not in event
        assert "payload_json" not in event
        assert "target_price" not in event

    asyncio.run(run())


def test_phase_5_execution_service_payload_lookup_boundary_is_documented() -> None:
    doc = CONTRACT_DOC.read_text(encoding="utf-8")

    assert "`bot_signal.persisted.v1` is a notification event, not the payload transport" in doc
    assert "_bot_platform.bot_signals.payload_json" in doc
    assert "by `signal_id`" in doc


def _build_signal(
    *,
    payload: dict[str, object],
    signal_type: BotSignalType,
    side: BotSignalSide,
) -> BotSignal:
    return BotSignal.build(
        instance_id="instance-1",
        module_id="spot_grid",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=signal_type,
        side=side,
        confidence=Decimal("0.80"),
        reason="phase_5_contract",
        payload_schema=POSITION_INTENT_PAYLOAD_SCHEMA,
        payload_schema_version=POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
        payload=payload,
    )
