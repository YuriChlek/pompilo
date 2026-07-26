from __future__ import annotations

from dataclasses import dataclass

from trade_execution_service.domain import (
    BotSignalPersistedEvent,
    ExchangeExecutionAdapter,
    ExecutionAccountResolver,
    ExecutionAuditLog,
    ExecutionControlPolicy,
    ExecutionDecision,
    ExecutionDecisionStatus,
    ExecutionIdempotencyStore,
    ExecutionIntent,
    ExecutionIntentType,
    ExecutionStatusPublisher,
    PersistedBotSignalRef,
    SignalPayloadStore,
)

PERSISTED_BOT_SIGNAL_EVENT_TYPE = "bot_signal.persisted.v1"


@dataclass(frozen=True, slots=True)
class ExecuteSignalCommand:
    """Command to process one persisted bot signal."""

    signal_id: str
    signal_key: str | None = None
    correlation_id: str | None = None


class AllowExecutionControlPolicy:
    """Default policy used by tests/scaffold when no runtime controls are configured."""

    async def is_global_kill_switch_enabled(self) -> bool:
        return False

    async def is_bot_or_symbol_paused(self, *, signal: PersistedBotSignalRef) -> bool:
        _ = signal
        return False


class InMemoryExecutionIdempotencyStore:
    """Process-local idempotency store for scaffold and unit tests."""

    def __init__(self) -> None:
        self._processed_keys: set[tuple[str, str | None]] = set()

    async def was_processed(self, *, signal_id: str, signal_key: str | None) -> bool:
        return (signal_id, signal_key) in self._processed_keys

    async def mark_processed(self, *, signal_id: str, signal_key: str | None) -> None:
        self._processed_keys.add((signal_id, signal_key))


class TradeExecutionService:
    """Application service that validates and routes persisted bot signals to exchanges."""

    def __init__(
        self,
        *,
        signal_store: SignalPayloadStore,
        account_resolver: ExecutionAccountResolver,
        exchange_adapters: dict[str, ExchangeExecutionAdapter],
        audit_log: ExecutionAuditLog,
        control_policy: ExecutionControlPolicy | None = None,
        idempotency_store: ExecutionIdempotencyStore | None = None,
        status_publisher: ExecutionStatusPublisher | None = None,
    ) -> None:
        self.signal_store = signal_store
        self.account_resolver = account_resolver
        self.exchange_adapters = exchange_adapters
        self.audit_log = audit_log
        self.control_policy = control_policy or AllowExecutionControlPolicy()
        self.idempotency_store = idempotency_store or InMemoryExecutionIdempotencyStore()
        self.status_publisher = status_publisher

    async def process_persisted_signal_event(self, event: BotSignalPersistedEvent) -> ExecutionDecision:
        """Process one metadata-only Bot Platform persisted signal event."""

        if event.event_type != PERSISTED_BOT_SIGNAL_EVENT_TYPE:
            decision = ExecutionDecision(
                signal_id=event.signal_id,
                status=ExecutionDecisionStatus.REJECTED,
                reason=f"unsupported_signal_event_type:{event.event_type}",
            )
            await self._record_decision(decision)
            return decision

        command = ExecuteSignalCommand(
            signal_id=event.signal_id,
            signal_key=event.signal_key,
            correlation_id=event.correlation_id,
        )
        return await self.execute_signal(command)

    async def execute_signal(self, command: ExecuteSignalCommand) -> ExecutionDecision:
        if await self.idempotency_store.was_processed(
            signal_id=command.signal_id,
            signal_key=command.signal_key,
        ):
            decision = ExecutionDecision(
                signal_id=command.signal_id,
                status=ExecutionDecisionStatus.SKIPPED,
                reason="duplicate_signal_already_processed",
            )
            await self._record_decision(decision)
            return decision

        signal, payload = await self.signal_store.load_signal_payload(signal_id=command.signal_id)
        signal_key = command.signal_key or signal.signal_key
        if signal_key != command.signal_key and await self.idempotency_store.was_processed(
            signal_id=signal.signal_id,
            signal_key=signal_key,
        ):
            decision = ExecutionDecision(
                signal_id=signal.signal_id,
                status=ExecutionDecisionStatus.SKIPPED,
                reason="duplicate_signal_already_processed",
            )
            await self._record_decision(decision)
            return decision

        if await self.control_policy.is_global_kill_switch_enabled():
            decision = ExecutionDecision(
                signal_id=signal.signal_id,
                status=ExecutionDecisionStatus.REJECTED,
                reason="global_execution_kill_switch_enabled",
            )
            await self._mark_and_record(signal=signal, signal_key=signal_key, decision=decision)
            return decision

        if await self.control_policy.is_bot_or_symbol_paused(signal=signal):
            decision = ExecutionDecision(
                signal_id=signal.signal_id,
                status=ExecutionDecisionStatus.SKIPPED,
                reason="bot_or_symbol_execution_paused",
            )
            await self._mark_and_record(signal=signal, signal_key=signal_key, decision=decision)
            return decision

        intent = _intent_from_payload(signal, payload)
        if intent.intent_type in {ExecutionIntentType.HOLD, ExecutionIntentType.ALERT}:
            decision = ExecutionDecision(
                signal_id=signal.signal_id,
                status=ExecutionDecisionStatus.SKIPPED,
                reason=f"intent_type_{intent.intent_type.value}_is_not_executable",
            )
            await self._mark_and_record(signal=signal, signal_key=signal_key, decision=decision)
            return decision

        account = await self.account_resolver.resolve_account(signal=signal)
        adapter = self.exchange_adapters.get(account.exchange_id.value)
        if adapter is None:
            decision = ExecutionDecision(
                signal_id=signal.signal_id,
                status=ExecutionDecisionStatus.REJECTED,
                reason=f"exchange_adapter_not_configured:{account.exchange_id.value}",
                exchange_id=account.exchange_id,
            )
            await self._mark_and_record(signal=signal, signal_key=signal_key, decision=decision)
            return decision

        await adapter.get_balances(account=account)
        await adapter.get_position(account=account, symbol=intent.symbol)
        await adapter.get_constraints(account=account, symbol=intent.symbol)

        decision = await adapter.execute(account=account, intent=intent)
        await self._mark_and_record(signal=signal, signal_key=signal_key, decision=decision)
        return decision

    async def _mark_and_record(
        self,
        *,
        signal: PersistedBotSignalRef,
        signal_key: str | None,
        decision: ExecutionDecision,
    ) -> None:
        await self.idempotency_store.mark_processed(signal_id=signal.signal_id, signal_key=signal_key)
        await self._record_decision(decision)

    async def _record_decision(self, decision: ExecutionDecision) -> None:
        await self.audit_log.append_decision(decision=decision)
        if self.status_publisher is not None:
            await self.status_publisher.publish_status(decision=decision)


def _intent_from_payload(signal: PersistedBotSignalRef, payload: dict[str, object]) -> ExecutionIntent:
    """Build a minimal execution intent from a signal payload.

    Full risk, stale-snapshot, and sizing checks remain exchange-service concerns and
    must be applied before concrete venue order placement.
    """

    intent_type = ExecutionIntentType(str(payload.get("intent_type", "alert")))
    return ExecutionIntent(
        signal=signal,
        intent_type=intent_type,
        symbol=str(payload.get("symbol") or signal.symbol).upper(),
        side=None,
        order_type=None,
        target_price=None,
        suggested_quote_notional=None,
        max_quote_notional=None,
        reason_codes=tuple(str(item) for item in payload.get("reason_codes", ()) if str(item)),
        payload=payload,
    )
