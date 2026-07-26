from __future__ import annotations

from decimal import Decimal

from trade_execution_service.domain import (
    BalanceSnapshot,
    ExchangeId,
    ExecutionAccountRef,
    ExecutionDecision,
    ExecutionDecisionStatus,
    ExecutionIntent,
    PositionSnapshot,
    VenueConstraints,
)


class OkxExecutionAdapter:
    """Placeholder OKX adapter."""

    async def get_balances(self, *, account: ExecutionAccountRef) -> tuple[BalanceSnapshot, ...]:
        _ = account
        return ()

    async def get_position(self, *, account: ExecutionAccountRef, symbol: str) -> PositionSnapshot:
        return PositionSnapshot(
            account_id=account.account_id,
            exchange_id=ExchangeId.OKX,
            symbol=symbol.upper(),
            base_quantity=Decimal("0"),
            average_entry_price=None,
        )

    async def get_constraints(self, *, account: ExecutionAccountRef, symbol: str) -> VenueConstraints:
        _ = account
        return VenueConstraints(
            exchange_id=ExchangeId.OKX,
            symbol=symbol.upper(),
            min_quantity=Decimal("0"),
            min_notional=Decimal("0"),
            quantity_step=Decimal("0.00000001"),
            price_tick=Decimal("0.00000001"),
        )

    async def execute(self, *, account: ExecutionAccountRef, intent: ExecutionIntent) -> ExecutionDecision:
        return ExecutionDecision(
            signal_id=intent.signal.signal_id,
            status=ExecutionDecisionStatus.REJECTED,
            reason="okx_order_placement_not_implemented",
            exchange_id=account.exchange_id,
        )
