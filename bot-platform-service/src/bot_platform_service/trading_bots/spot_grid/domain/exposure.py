from __future__ import annotations

from decimal import Decimal, localcontext

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    PortfolioContext,
    PortfolioExposureSnapshot,
    SymbolExposureSnapshot,
)


def compute_exposure(portfolio_context: PortfolioContext) -> PortfolioExposureSnapshot:
    """Compute deterministic exposure from platform-supplied portfolio context."""

    per_symbol = _symbol_exposures(portfolio_context)
    gross_quote_notional = sum((exposure.quote_notional for exposure in per_symbol), Decimal("0"))
    return PortfolioExposureSnapshot(
        total_equity=portfolio_context.total_equity,
        available_quote=portfolio_context.available_quote,
        gross_quote_notional=gross_quote_notional,
        gross_exposure_fraction=_ratio_or_none(gross_quote_notional, portfolio_context.total_equity),
        available_quote_fraction=_ratio_or_none(portfolio_context.available_quote, portfolio_context.total_equity),
        per_symbol=per_symbol,
    )


def _symbol_exposures(portfolio_context: PortfolioContext) -> tuple[SymbolExposureSnapshot, ...]:
    grouped: dict[str, dict[str, Decimal | int]] = {}
    for position in portfolio_context.positions:
        current = grouped.setdefault(
            position.symbol,
            {
                "position_count": 0,
                "base_quantity": Decimal("0"),
                "quote_notional": Decimal("0"),
            },
        )
        current["position_count"] = int(current["position_count"]) + 1
        current["base_quantity"] = Decimal(current["base_quantity"]) + position.base_quantity
        current["quote_notional"] = Decimal(current["quote_notional"]) + position.quote_notional

    return tuple(
        SymbolExposureSnapshot(
            symbol=symbol,
            position_count=int(values["position_count"]),
            base_quantity=Decimal(values["base_quantity"]),
            quote_notional=Decimal(values["quote_notional"]),
            exposure_fraction=_ratio_or_none(Decimal(values["quote_notional"]), portfolio_context.total_equity),
        )
        for symbol, values in sorted(grouped.items())
    )


def _ratio_or_none(value: Decimal, baseline: Decimal) -> Decimal | None:
    if baseline <= Decimal("0"):
        return None
    with localcontext() as context:
        context.prec = 34
        return +(value / baseline)


__all__ = ["compute_exposure"]
