from __future__ import annotations

from decimal import Decimal

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    PortfolioAllocationSnapshot,
    PositionContext,
    PriceBand,
    StrategyGuardSnapshot,
    StrategyRiskLimits,
    TargetExecutionIntent,
    TargetIntent,
    TargetIntentType,
    UnderwaterPositionSnapshot,
    UnderwaterStateSnapshot,
)
from bot_platform_service.trading_bots.spot_grid.domain.underwater import RECOVERY_ALLOWED_REGIMES


def build_recovery_averaging_intents(
    *,
    symbol: str,
    timeframe: str,
    reference_price: Decimal,
    price_band: PriceBand,
    regime: MarketRegime,
    underwater_state: UnderwaterStateSnapshot | None,
    allocation: PortfolioAllocationSnapshot | None,
    indicators: IndicatorSnapshot | None,
    rsi_buy_allowed: bool,
) -> tuple[TargetIntent, ...]:
    """Build execution-neutral recovery averaging buy intents."""

    if not _recovery_guardrails_pass(
        regime=regime,
        underwater_state=underwater_state,
        allocation=allocation,
        rsi_buy_allowed=rsi_buy_allowed,
    ):
        return ()

    risk = StrategyRiskLimits(
        max_position_fraction=allocation.max_position_fraction,
        suggested_quote_notional=allocation.suggested_quote_notional,
        max_quote_notional=allocation.max_quote_notional,
    )
    return tuple(
        TargetIntent(
            intent_type=TargetIntentType.OPEN_POSITION,
            execution_intent=TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE,
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            side=GridLevelSide.BUY,
            target_price=reference_price,
            reference_price=reference_price,
            price_band=price_band,
            risk=risk,
            guards=StrategyGuardSnapshot(
                buy_allowed=True,
                sell_allowed=False,
                rsi14=indicators.rsi14 if indicators is not None else None,
                atr14=indicators.atr14 if indicators is not None else None,
            ),
            reason_codes=_recovery_reason_codes(position),
            position=_position_context(position),
            metadata={
                "recovery_strategy": "averaging",
                "unrealized_pnl_fraction": str(position.unrealized_pnl_fraction)
                if position.unrealized_pnl_fraction is not None
                else None,
                "budget_source": "portfolio_allocator",
            },
        )
        for position in underwater_state.positions
        if position.recovery_eligible
    )


def recovery_averaging_diagnostics(
    *,
    regime: MarketRegime,
    underwater_state: UnderwaterStateSnapshot | None,
    allocation: PortfolioAllocationSnapshot | None,
    rsi_buy_allowed: bool,
    intent_count: int,
) -> dict[str, object]:
    """Explain why recovery averaging intents were emitted or blocked."""

    budget_available = _budget_available(allocation)
    reason_codes: list[str] = []
    if underwater_state is None:
        reason_codes.append("underwater_state_unavailable")
    elif underwater_state.reason_codes:
        reason_codes.extend(underwater_state.reason_codes)
    if regime not in RECOVERY_ALLOWED_REGIMES:
        reason_codes.append("recovery_averaging_regime_block")
    if not rsi_buy_allowed:
        reason_codes.append("recovery_averaging_rsi_buy_block")
    if not budget_available:
        reason_codes.append("recovery_averaging_budget_block")
    if intent_count:
        reason_codes.append("recovery_averaging_intent_created")
    elif "recovery_not_eligible" not in reason_codes:
        reason_codes.append("recovery_not_eligible")

    return {
        "intent_count": intent_count,
        "budget_available": budget_available,
        "blocked_regime": regime not in RECOVERY_ALLOWED_REGIMES,
        "rsi_buy_allowed": rsi_buy_allowed,
        "reason_codes": tuple(dict.fromkeys(reason_codes)),
    }


def _recovery_guardrails_pass(
    *,
    regime: MarketRegime,
    underwater_state: UnderwaterStateSnapshot | None,
    allocation: PortfolioAllocationSnapshot | None,
    rsi_buy_allowed: bool,
) -> bool:
    return (
        underwater_state is not None
        and underwater_state.recovery_eligible_count > 0
        and regime in RECOVERY_ALLOWED_REGIMES
        and rsi_buy_allowed
        and _budget_available(allocation)
    )


def _budget_available(allocation: PortfolioAllocationSnapshot | None) -> bool:
    return (
        allocation is not None
        and allocation.suggested_quote_notional > Decimal("0")
        and allocation.max_quote_notional > Decimal("0")
    )


def _recovery_reason_codes(position: UnderwaterPositionSnapshot) -> tuple[str, ...]:
    return (
        "recovery_averaging_buy",
        "budget_available",
        *(reason for reason in position.reason_codes if reason != "recovery_not_eligible"),
    )


def _position_context(position: UnderwaterPositionSnapshot) -> PositionContext:
    return PositionContext(
        symbol=position.symbol,
        base_quantity=position.base_quantity,
        quote_notional=position.quote_notional,
        cost_basis=position.cost_basis,
    )


__all__ = ["build_recovery_averaging_intents", "recovery_averaging_diagnostics"]
