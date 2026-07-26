from __future__ import annotations

from decimal import Decimal

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    PositionContext,
    PriceBand,
    StrategyGuardSnapshot,
    StrategyRiskLimits,
    TargetExecutionIntent,
    TargetIntent,
    TargetIntentType,
)

DE_RISK_ALLOWED_REGIMES = frozenset((MarketRegime.RISK_OFF,))


def build_de_risk_intents(
    *,
    symbol: str,
    timeframe: str,
    reference_price: Decimal,
    price_band: PriceBand,
    regime: MarketRegime,
    max_position_fraction: Decimal,
    position_context: PositionContext | None,
    indicators: IndicatorSnapshot | None,
    rsi_sell_allowed: bool,
) -> tuple[TargetIntent, ...]:
    """Build execution-neutral sell intents that reduce existing position risk."""

    if not _de_risk_guardrails_pass(regime=regime, position_context=position_context):
        return ()

    return (
        TargetIntent(
            intent_type=TargetIntentType.CLOSE_POSITION,
            execution_intent=TargetExecutionIntent.DE_RISK_CANDIDATE,
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            side=GridLevelSide.SELL,
            target_price=reference_price,
            reference_price=reference_price,
            price_band=price_band,
            risk=StrategyRiskLimits(max_position_fraction=max_position_fraction),
            guards=StrategyGuardSnapshot(
                buy_allowed=False,
                sell_allowed=True,
                rsi14=indicators.rsi14 if indicators is not None else None,
                atr14=indicators.atr14 if indicators is not None else None,
                no_loss_required=False,
                no_loss_passed=None,
            ),
            reason_codes=_de_risk_reason_codes(regime=regime, rsi_sell_allowed=rsi_sell_allowed),
            position=position_context,
            metadata={
                "de_risk_strategy": "risk_reduction",
                "execution_risk_recheck_required": True,
                "rsi_sell_rule_bypassed": not rsi_sell_allowed,
            },
        ),
    )


def de_risk_diagnostics(
    *,
    regime: MarketRegime,
    position_context: PositionContext | None,
    rsi_sell_allowed: bool,
    intent_count: int,
) -> dict[str, object]:
    """Explain why de-risk intents were emitted or blocked."""

    has_open_position = _has_open_position(position_context)
    reason_codes: list[str] = []
    if regime not in DE_RISK_ALLOWED_REGIMES:
        reason_codes.append("de_risk_regime_block")
    if not has_open_position:
        reason_codes.append("de_risk_position_missing")
    if intent_count:
        reason_codes.append("de_risk_sell_intent_created")
    else:
        reason_codes.append("de_risk_not_eligible")
    if intent_count and not rsi_sell_allowed:
        reason_codes.append("de_risk_rsi_sell_rule_bypassed")

    return {
        "intent_count": intent_count,
        "blocked_regime": regime not in DE_RISK_ALLOWED_REGIMES,
        "has_open_position": has_open_position,
        "rsi_sell_allowed": rsi_sell_allowed,
        "reason_codes": tuple(dict.fromkeys(reason_codes)),
    }


def _de_risk_guardrails_pass(*, regime: MarketRegime, position_context: PositionContext | None) -> bool:
    return regime in DE_RISK_ALLOWED_REGIMES and _has_open_position(position_context)


def _has_open_position(position_context: PositionContext | None) -> bool:
    return (
        position_context is not None
        and position_context.base_quantity > Decimal("0")
        and position_context.quote_notional > Decimal("0")
    )


def _de_risk_reason_codes(*, regime: MarketRegime, rsi_sell_allowed: bool) -> tuple[str, ...]:
    reasons = [
        "de_risk_sell",
        f"{regime.value}_de_risk",
        "position_exposure_detected",
        "execution_risk_recheck_required",
    ]
    if not rsi_sell_allowed:
        reasons.append("rsi_sell_rule_bypassed")
    return tuple(reasons)


__all__ = ["DE_RISK_ALLOWED_REGIMES", "build_de_risk_intents", "de_risk_diagnostics"]
