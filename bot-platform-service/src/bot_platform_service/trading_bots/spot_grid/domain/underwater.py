from __future__ import annotations

from decimal import Decimal, localcontext

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    MarketRegime,
    PortfolioContext,
    UnderwaterPositionSnapshot,
    UnderwaterStateSnapshot,
)

RECOVERY_ALLOWED_REGIMES = frozenset((MarketRegime.RANGE, MarketRegime.UPTREND))


def detect_underwater_state(
    *,
    portfolio_context: PortfolioContext,
    symbol: str,
    reference_price: Decimal,
    regime: MarketRegime,
    entry_block_reasons: tuple[str, ...],
) -> UnderwaterStateSnapshot:
    """Detect underwater positions and recovery eligibility without creating intents."""

    positions = tuple(
        _position_state(
            position=position,
            reference_price=reference_price,
            regime=regime,
            entry_block_reasons=entry_block_reasons,
        )
        for position in portfolio_context.positions
        if position.symbol == symbol.upper()
    )
    underwater_count = sum(1 for position in positions if position.underwater)
    recovery_eligible_count = sum(1 for position in positions if position.recovery_eligible)
    reason_codes = _summary_reason_codes(
        positions=positions,
        underwater_count=underwater_count,
        recovery_eligible_count=recovery_eligible_count,
    )
    return UnderwaterStateSnapshot(
        reference_price=reference_price,
        regime=regime,
        positions=positions,
        underwater_count=underwater_count,
        recovery_eligible_count=recovery_eligible_count,
        reason_codes=reason_codes,
    )


def _position_state(
    *,
    position,
    reference_price: Decimal,
    regime: MarketRegime,
    entry_block_reasons: tuple[str, ...],
) -> UnderwaterPositionSnapshot:
    pnl_fraction = _pnl_fraction(reference_price=reference_price, cost_basis=position.cost_basis)
    underwater = pnl_fraction is not None and pnl_fraction < Decimal("0")
    reason_codes = _position_reason_codes(
        has_position=position.base_quantity > Decimal("0"),
        has_cost_basis=position.cost_basis is not None,
        underwater=underwater,
        regime=regime,
        entry_block_reasons=entry_block_reasons,
    )
    recovery_eligible = "recovery_eligible" in reason_codes
    return UnderwaterPositionSnapshot(
        symbol=position.symbol,
        base_quantity=position.base_quantity,
        quote_notional=position.quote_notional,
        cost_basis=position.cost_basis,
        reference_price=reference_price,
        unrealized_pnl_fraction=pnl_fraction,
        underwater=underwater,
        recovery_eligible=recovery_eligible,
        reason_codes=reason_codes,
    )


def _position_reason_codes(
    *,
    has_position: bool,
    has_cost_basis: bool,
    underwater: bool,
    regime: MarketRegime,
    entry_block_reasons: tuple[str, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not has_position:
        reasons.append("no_open_position")
    if not has_cost_basis:
        reasons.append("cost_basis_unknown")
    if not underwater:
        reasons.append("position_not_underwater")
    else:
        reasons.append("position_underwater")
    if regime not in RECOVERY_ALLOWED_REGIMES:
        reasons.append("recovery_regime_block")
    if entry_block_reasons:
        reasons.append("recovery_entry_guard_block")
    if has_position and has_cost_basis and underwater and regime in RECOVERY_ALLOWED_REGIMES and not entry_block_reasons:
        reasons.append("recovery_eligible")
    else:
        reasons.append("recovery_not_eligible")
    return tuple(reasons)


def _summary_reason_codes(
    *,
    positions: tuple[UnderwaterPositionSnapshot, ...],
    underwater_count: int,
    recovery_eligible_count: int,
) -> tuple[str, ...]:
    if not positions:
        return ("no_symbol_position", "recovery_not_eligible")
    reasons: list[str] = []
    reasons.append("underwater_positions_detected" if underwater_count else "no_underwater_positions")
    reasons.append("recovery_eligible" if recovery_eligible_count else "recovery_not_eligible")
    return tuple(reasons)


def _pnl_fraction(*, reference_price: Decimal, cost_basis: Decimal | None) -> Decimal | None:
    if cost_basis is None or cost_basis <= Decimal("0"):
        return None
    with localcontext() as context:
        context.prec = 34
        return +((reference_price - cost_basis) / cost_basis)


__all__ = ["RECOVERY_ALLOWED_REGIMES", "detect_underwater_state"]
