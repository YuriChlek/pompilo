from __future__ import annotations

from decimal import Decimal, localcontext

from bot_platform_service.trading_bots.spot_grid.domain.exposure import compute_exposure
from bot_platform_service.trading_bots.spot_grid.domain.models import (
    PortfolioAllocationSnapshot,
    PortfolioContext,
    normalize_spot_grid_symbol,
)


def allocate_portfolio_budget(
    *,
    portfolio_context: PortfolioContext,
    symbol: str,
    max_position_fraction: Decimal,
    candidate_count: int,
) -> PortfolioAllocationSnapshot:
    """Calculate execution-neutral per-symbol and portfolio budget caps."""

    normalized_symbol = normalize_spot_grid_symbol(symbol)
    exposure = compute_exposure(portfolio_context)
    symbol_exposure = next(
        (candidate for candidate in exposure.per_symbol if candidate.symbol == normalized_symbol),
        None,
    )
    symbol_current_quote_notional = (
        symbol_exposure.quote_notional if symbol_exposure is not None else Decimal("0")
    )
    symbol_max_quote_notional = _non_negative(portfolio_context.total_equity * max_position_fraction)
    symbol_remaining_quote_notional = _non_negative(symbol_max_quote_notional - symbol_current_quote_notional)
    portfolio_remaining_quote_notional = _non_negative(portfolio_context.available_quote)
    max_quote_notional = min(symbol_remaining_quote_notional, portfolio_remaining_quote_notional)
    normalized_candidate_count = max(candidate_count, 0)
    suggested_quote_notional = _suggested_quote_notional(
        max_quote_notional=max_quote_notional,
        candidate_count=normalized_candidate_count,
    )

    return PortfolioAllocationSnapshot(
        symbol=normalized_symbol,
        max_position_fraction=max_position_fraction,
        symbol_current_quote_notional=symbol_current_quote_notional,
        symbol_max_quote_notional=symbol_max_quote_notional,
        symbol_remaining_quote_notional=symbol_remaining_quote_notional,
        portfolio_available_quote=portfolio_context.available_quote,
        portfolio_remaining_quote_notional=portfolio_remaining_quote_notional,
        max_quote_notional=max_quote_notional,
        suggested_quote_notional=suggested_quote_notional,
        candidate_count=normalized_candidate_count,
    )


def _suggested_quote_notional(*, max_quote_notional: Decimal, candidate_count: int) -> Decimal:
    if max_quote_notional <= Decimal("0") or candidate_count <= 0:
        return Decimal("0")
    with localcontext() as context:
        context.prec = 34
        return +(max_quote_notional / Decimal(candidate_count))


def _non_negative(value: Decimal) -> Decimal:
    return max(value, Decimal("0"))


__all__ = ["allocate_portfolio_budget"]
