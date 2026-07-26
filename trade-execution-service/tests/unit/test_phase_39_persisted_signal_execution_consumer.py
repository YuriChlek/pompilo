from __future__ import annotations

import asyncio
from decimal import Decimal

from trade_execution_service.application import PERSISTED_BOT_SIGNAL_EVENT_TYPE, TradeExecutionService
from trade_execution_service.domain import (
    BalanceSnapshot,
    BotSignalPersistedEvent,
    ExchangeId,
    ExecutionAccountRef,
    ExecutionDecision,
    ExecutionDecisionStatus,
    ExecutionIntent,
    PersistedBotSignalRef,
    PositionSnapshot,
    VenueConstraints,
)


def test_persisted_signal_event_loads_full_payload_before_execution() -> None:
    signal_store = _SignalStore(payload={"intent_type": "open_position", "symbol": "ethusdt"})
    adapter = _RecordingExchangeAdapter()
    audit_log = _AuditLog()
    status_publisher = _StatusPublisher()
    service = _service(
        signal_store=signal_store,
        adapter=adapter,
        audit_log=audit_log,
        status_publisher=status_publisher,
    )

    decision = asyncio.run(
        service.process_persisted_signal_event(
            BotSignalPersistedEvent(
                event_type=PERSISTED_BOT_SIGNAL_EVENT_TYPE,
                signal_id="sig-39",
                signal_key="signal-key-39",
                correlation_id="corr-39",
            )
        )
    )

    assert signal_store.loaded_signal_ids == ["sig-39"]
    assert adapter.executed_intents[0].payload == {"intent_type": "open_position", "symbol": "ethusdt"}
    assert adapter.balance_checks == ["account-for-instance-1"]
    assert adapter.position_checks == ["ETHUSDT"]
    assert adapter.constraint_checks == ["ETHUSDT"]
    assert decision.status is ExecutionDecisionStatus.PLACED
    assert audit_log.decisions == [decision]
    assert status_publisher.decisions == [decision]


def test_duplicate_persisted_signal_event_is_idempotently_skipped() -> None:
    signal_store = _SignalStore(payload={"intent_type": "open_position", "symbol": "ETHUSDT"})
    adapter = _RecordingExchangeAdapter()
    audit_log = _AuditLog()
    service = _service(signal_store=signal_store, adapter=adapter, audit_log=audit_log)
    event = BotSignalPersistedEvent(
        event_type=PERSISTED_BOT_SIGNAL_EVENT_TYPE,
        signal_id="sig-duplicate",
        signal_key="duplicate-key",
    )

    first_decision = asyncio.run(service.process_persisted_signal_event(event))
    duplicate_decision = asyncio.run(service.process_persisted_signal_event(event))

    assert first_decision.status is ExecutionDecisionStatus.PLACED
    assert duplicate_decision.status is ExecutionDecisionStatus.SKIPPED
    assert duplicate_decision.reason == "duplicate_signal_already_processed"
    assert signal_store.loaded_signal_ids == ["sig-duplicate"]
    assert len(adapter.executed_intents) == 1
    assert audit_log.decisions == [first_decision, duplicate_decision]


def test_global_kill_switch_blocks_execution_after_payload_lookup() -> None:
    signal_store = _SignalStore(payload={"intent_type": "open_position", "symbol": "ETHUSDT"})
    adapter = _RecordingExchangeAdapter()
    service = _service(
        signal_store=signal_store,
        adapter=adapter,
        control_policy=_ControlPolicy(global_kill=True),
    )

    decision = asyncio.run(
        service.process_persisted_signal_event(
            BotSignalPersistedEvent(
                event_type=PERSISTED_BOT_SIGNAL_EVENT_TYPE,
                signal_id="sig-kill",
                signal_key="kill-key",
            )
        )
    )

    assert signal_store.loaded_signal_ids == ["sig-kill"]
    assert decision.status is ExecutionDecisionStatus.REJECTED
    assert decision.reason == "global_execution_kill_switch_enabled"
    assert adapter.executed_intents == []


def test_bot_or_symbol_pause_blocks_execution_consumer() -> None:
    signal_store = _SignalStore(payload={"intent_type": "open_position", "symbol": "ETHUSDT"})
    adapter = _RecordingExchangeAdapter()
    service = _service(
        signal_store=signal_store,
        adapter=adapter,
        control_policy=_ControlPolicy(paused=True),
    )

    decision = asyncio.run(
        service.process_persisted_signal_event(
            BotSignalPersistedEvent(
                event_type=PERSISTED_BOT_SIGNAL_EVENT_TYPE,
                signal_id="sig-paused",
                signal_key="paused-key",
            )
        )
    )

    assert decision.status is ExecutionDecisionStatus.SKIPPED
    assert decision.reason == "bot_or_symbol_execution_paused"
    assert adapter.executed_intents == []


def test_consumer_rejects_non_stable_signal_stream_event() -> None:
    signal_store = _SignalStore(payload={"intent_type": "open_position", "symbol": "ETHUSDT"})
    adapter = _RecordingExchangeAdapter()
    service = _service(signal_store=signal_store, adapter=adapter)

    decision = asyncio.run(
        service.process_persisted_signal_event(
            BotSignalPersistedEvent(event_type="bot_signal.preview.v1", signal_id="sig-preview")
        )
    )

    assert decision.status is ExecutionDecisionStatus.REJECTED
    assert decision.reason == "unsupported_signal_event_type:bot_signal.preview.v1"
    assert signal_store.loaded_signal_ids == []
    assert adapter.executed_intents == []


def _service(
    *,
    signal_store: _SignalStore,
    adapter: _RecordingExchangeAdapter,
    audit_log: _AuditLog | None = None,
    control_policy: _ControlPolicy | None = None,
    status_publisher: _StatusPublisher | None = None,
) -> TradeExecutionService:
    return TradeExecutionService(
        signal_store=signal_store,
        account_resolver=_AccountResolver(),
        exchange_adapters={"bybit": adapter},
        audit_log=audit_log or _AuditLog(),
        control_policy=control_policy,
        status_publisher=status_publisher,
    )


class _SignalStore:
    def __init__(self, *, payload: dict[str, object]) -> None:
        self.loaded_signal_ids: list[str] = []
        self.payload = payload

    async def load_signal_payload(self, *, signal_id: str):
        self.loaded_signal_ids.append(signal_id)
        return (
            PersistedBotSignalRef(
                signal_id=signal_id,
                signal_key="persisted-signal-key",
                instance_id="instance-1",
                module_id="spot_grid",
                symbol="ETHUSDT",
                timeframe="1m",
                snapshot_id="snapshot-1",
                payload_schema="spot_grid.position_intent",
                payload_schema_version=1,
                payload_hash="hash-1",
            ),
            self.payload,
        )


class _AccountResolver:
    async def resolve_account(self, *, signal: PersistedBotSignalRef) -> ExecutionAccountRef:
        return ExecutionAccountRef(
            tenant_id="tenant-1",
            user_id="user-1",
            account_id=f"account-for-{signal.instance_id}",
            exchange_id=ExchangeId.BYBIT,
            secret_ref="secret/bybit/account-1",
        )


class _RecordingExchangeAdapter:
    def __init__(self) -> None:
        self.balance_checks: list[str] = []
        self.position_checks: list[str] = []
        self.constraint_checks: list[str] = []
        self.executed_intents: list[ExecutionIntent] = []

    async def get_balances(self, *, account: ExecutionAccountRef) -> tuple[BalanceSnapshot, ...]:
        self.balance_checks.append(account.account_id)
        return (
            BalanceSnapshot(
                account_id=account.account_id,
                exchange_id=account.exchange_id,
                asset="USDT",
                available=Decimal("100"),
                total=Decimal("100"),
            ),
        )

    async def get_position(self, *, account: ExecutionAccountRef, symbol: str) -> PositionSnapshot:
        self.position_checks.append(symbol)
        return PositionSnapshot(
            account_id=account.account_id,
            exchange_id=account.exchange_id,
            symbol=symbol,
            base_quantity=Decimal("0"),
            average_entry_price=None,
        )

    async def get_constraints(self, *, account: ExecutionAccountRef, symbol: str) -> VenueConstraints:
        self.constraint_checks.append(symbol)
        return VenueConstraints(
            exchange_id=account.exchange_id,
            symbol=symbol,
            min_quantity=Decimal("0.001"),
            min_notional=Decimal("5"),
            quantity_step=Decimal("0.001"),
            price_tick=Decimal("0.01"),
        )

    async def execute(self, *, account: ExecutionAccountRef, intent: ExecutionIntent) -> ExecutionDecision:
        self.executed_intents.append(intent)
        return ExecutionDecision(
            signal_id=intent.signal.signal_id,
            status=ExecutionDecisionStatus.PLACED,
            reason="order_placed",
            exchange_id=account.exchange_id,
            venue_order_id="venue-order-1",
        )


class _AuditLog:
    def __init__(self) -> None:
        self.decisions: list[ExecutionDecision] = []

    async def append_decision(self, *, decision: ExecutionDecision) -> None:
        self.decisions.append(decision)


class _StatusPublisher:
    def __init__(self) -> None:
        self.decisions: list[ExecutionDecision] = []

    async def publish_status(self, *, decision: ExecutionDecision) -> None:
        self.decisions.append(decision)


class _ControlPolicy:
    def __init__(self, *, global_kill: bool = False, paused: bool = False) -> None:
        self.global_kill = global_kill
        self.paused = paused

    async def is_global_kill_switch_enabled(self) -> bool:
        return self.global_kill

    async def is_bot_or_symbol_paused(self, *, signal: PersistedBotSignalRef) -> bool:
        _ = signal
        return self.paused
