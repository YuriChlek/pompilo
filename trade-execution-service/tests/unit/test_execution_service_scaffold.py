from __future__ import annotations

import asyncio

from trade_execution_service.application import ExecuteSignalCommand, TradeExecutionService
from trade_execution_service.domain import (
    ExchangeId,
    ExecutionAccountRef,
    ExecutionDecision,
    ExecutionDecisionStatus,
    PersistedBotSignalRef,
)
from trade_execution_service.infrastructure.exchanges import build_exchange_adapters


def test_exchange_registry_contains_supported_placeholder_adapters() -> None:
    adapters = build_exchange_adapters()

    assert set(adapters) == {"binance", "bybit", "okx"}


def test_worker_scaffold_rejects_unimplemented_exchange_execution() -> None:
    signal_store = _SignalStore()
    service = TradeExecutionService(
        signal_store=signal_store,
        account_resolver=_AccountResolver(),
        exchange_adapters=build_exchange_adapters(),
        audit_log=_AuditLog(),
    )

    decision = asyncio.run(service.execute_signal(ExecuteSignalCommand(signal_id="sig-1")))

    assert signal_store.loaded_signal_ids == ["sig-1"]
    assert decision.status is ExecutionDecisionStatus.REJECTED
    assert decision.exchange_id is ExchangeId.BYBIT
    assert decision.reason == "bybit_order_placement_not_implemented"


def test_execution_service_contract_loads_full_payload_by_signal_id() -> None:
    signal_store = _SignalStore()
    service = TradeExecutionService(
        signal_store=signal_store,
        account_resolver=_AccountResolver(),
        exchange_adapters=build_exchange_adapters(),
        audit_log=_AuditLog(),
    )

    command = ExecuteSignalCommand(signal_id="sig-from-event", correlation_id="corr-1")
    decision = asyncio.run(service.execute_signal(command))

    assert signal_store.loaded_signal_ids == ["sig-from-event"]
    assert decision.signal_id == "sig-from-event"
    assert decision.status is ExecutionDecisionStatus.REJECTED


class _SignalStore:
    def __init__(self) -> None:
        self.loaded_signal_ids: list[str] = []

    async def load_signal_payload(self, *, signal_id: str):
        self.loaded_signal_ids.append(signal_id)
        return (
            PersistedBotSignalRef(
                signal_id=signal_id,
                signal_key="key",
                instance_id="instance-1",
                module_id="spot_grid",
                symbol="ETHUSDT",
                timeframe="1h",
                snapshot_id="snapshot-1",
                payload_schema="spot_grid.position_intent",
                payload_schema_version=1,
                payload_hash="hash",
            ),
            {
                "intent_type": "open_position",
                "symbol": "ETHUSDT",
                "reason_codes": ["range_buy"],
            },
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


class _AuditLog:
    def __init__(self) -> None:
        self.decisions: list[ExecutionDecision] = []

    async def append_decision(self, *, decision: ExecutionDecision) -> None:
        self.decisions.append(decision)
