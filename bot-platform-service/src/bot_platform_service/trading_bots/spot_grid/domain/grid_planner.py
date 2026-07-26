from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from bot_platform_service.trading_bots.spot_grid.domain.de_risk import build_de_risk_intents, de_risk_diagnostics
from bot_platform_service.trading_bots.spot_grid.domain.portfolio_allocator import allocate_portfolio_budget
from bot_platform_service.trading_bots.spot_grid.domain.recovery import (
    build_recovery_averaging_intents,
    recovery_averaging_diagnostics,
)
from bot_platform_service.trading_bots.spot_grid.domain.underwater import detect_underwater_state
from bot_platform_service.trading_bots.spot_grid.domain.models import (
    GridLevel,
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    MarketStructureSnapshot,
    PortfolioAllocationSnapshot,
    PositionContext,
    PortfolioContext,
    PriceBand,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlan,
    StrategyGuardSnapshot,
    StrategyRiskLimits,
    TargetExecutionIntent,
    TargetIntent,
    TargetIntentType,
)

DEFAULT_PRICE_QUANT = Decimal("0.00000001")
RSI_BUY_THRESHOLD = Decimal("35")
RSI_SELL_THRESHOLD = Decimal("65")
NO_LOSS_EXIT_MARKUP = Decimal("0.001")


class SpotGridPlanner:
    """Pure Decimal-based planner for platform-native Spot Grid runs."""

    def plan(
        self,
        *,
        symbol: str,
        timeframe: str,
        candles: tuple[SpotGridCandle, ...],
        config: SpotGridConfig,
        regime: MarketRegime = MarketRegime.RANGE,
        entry_block_reasons: tuple[str, ...] = (),
        indicators: IndicatorSnapshot | None = None,
        market_structure: MarketStructureSnapshot | None = None,
        portfolio_context: PortfolioContext | None = None,
        position_context: PositionContext | None = None,
    ) -> SpotGridPlan:
        """Build a balanced buy/sell grid from immutable platform candles."""
        if not candles:
            raise ValueError("candles must not be empty")
        if config.max_grid_levels <= 0:
            raise ValueError("max_grid_levels must be positive")

        reference_price = candles[-1].close
        range_low = min(candle.low for candle in candles)
        range_high = max(candle.high for candle in candles)
        if range_high <= range_low:
            band = max(reference_price * Decimal("0.01"), DEFAULT_PRICE_QUANT)
            range_low = reference_price - band
            range_high = reference_price + band

        reference_price = _quantize_price(reference_price)
        range_low = _quantize_price(range_low)
        range_high = _quantize_price(range_high)
        position_context = position_context or _position_for_symbol(portfolio_context, symbol=symbol)
        allocation = _allocation_for_symbol(
            portfolio_context=portfolio_context,
            symbol=symbol,
            max_position_fraction=config.max_position_fraction,
            candidate_count=config.max_grid_levels,
        )
        entry_allowed = regime not in (MarketRegime.DOWNTREND, MarketRegime.RISK_OFF) and not entry_block_reasons
        planner_name = "spot_grid_range_intent_grid"
        if regime is MarketRegime.UPTREND:
            levels, intents = _build_uptrend_grid(
                symbol=symbol,
                timeframe=timeframe,
                reference_price=reference_price,
                range_low=range_low,
                range_high=range_high,
                max_grid_levels=config.max_grid_levels,
                max_position_fraction=config.max_position_fraction,
                min_price_distance_fraction=config.min_price_distance_fraction,
                allocation=allocation,
                entry_allowed=entry_allowed,
                indicators=indicators,
                market_structure=market_structure,
            )
            planner_name = "spot_grid_uptrend_pullback_grid"
        else:
            levels, intents = _build_range_grid(
                symbol=symbol,
                timeframe=timeframe,
                reference_price=reference_price,
                range_low=range_low,
                range_high=range_high,
                max_grid_levels=config.max_grid_levels,
                max_position_fraction=config.max_position_fraction,
                min_price_distance_fraction=config.min_price_distance_fraction,
                allocation=allocation,
                regime=regime,
                entry_block_reasons=entry_block_reasons,
                indicators=indicators,
                position_context=position_context,
            )
        entry_allowed = entry_allowed and _rsi_buy_allowed(indicators)
        rsi_block_reasons = _rsi_block_reasons(indicators)
        no_loss_diagnostics = _no_loss_diagnostics(position_context)
        underwater_state = (
            detect_underwater_state(
                portfolio_context=portfolio_context,
                symbol=symbol,
                reference_price=reference_price,
                regime=regime,
                entry_block_reasons=entry_block_reasons,
            )
            if portfolio_context is not None
            else None
        )
        recovery_intents = build_recovery_averaging_intents(
            symbol=symbol,
            timeframe=timeframe,
            reference_price=reference_price,
            price_band=PriceBand(range_low=range_low, range_high=range_high),
            regime=regime,
            underwater_state=underwater_state,
            allocation=allocation,
            indicators=indicators,
            rsi_buy_allowed=_rsi_buy_allowed(indicators),
        )
        de_risk_intents = build_de_risk_intents(
            symbol=symbol,
            timeframe=timeframe,
            reference_price=reference_price,
            price_band=PriceBand(range_low=range_low, range_high=range_high),
            regime=regime,
            max_position_fraction=config.max_position_fraction,
            position_context=position_context,
            indicators=indicators,
            rsi_sell_allowed=_rsi_sell_allowed(indicators),
        )
        intents = (*intents, *recovery_intents, *de_risk_intents)
        price_distance_diagnostics = _price_distance_diagnostics(
            reference_price=reference_price,
            intents=tuple(intents),
            min_price_distance_fraction=config.min_price_distance_fraction,
        )
        allocation_diagnostics = allocation.to_payload() if allocation is not None else _empty_allocation_diagnostics(
            symbol=symbol,
            max_position_fraction=config.max_position_fraction,
            candidate_count=config.max_grid_levels,
        )

        return SpotGridPlan(
            symbol=symbol.upper(),
            timeframe=timeframe,
            reference_price=reference_price,
            range_low=range_low,
            range_high=range_high,
            levels=tuple(levels),
            diagnostics={
                "planner": planner_name,
                "candle_count": len(candles),
                "max_grid_levels": config.max_grid_levels,
                "max_position_fraction": str(config.max_position_fraction),
                "regime": regime.value,
                "entry_allowed": entry_allowed,
                "entry_block_reasons": entry_block_reasons,
                "intent_count": len(intents),
                "atr14": str(indicators.atr14) if indicators is not None and indicators.atr14 is not None else None,
                "rsi14": str(indicators.rsi14) if indicators is not None and indicators.rsi14 is not None else None,
                "rsi_buy_allowed": _rsi_buy_allowed(indicators),
                "rsi_sell_allowed": _rsi_sell_allowed(indicators),
                "rsi_block_reasons": rsi_block_reasons,
                "no_loss": no_loss_diagnostics,
                "price_distance": price_distance_diagnostics,
                "allocation": allocation_diagnostics,
                "recovery_averaging": recovery_averaging_diagnostics(
                    regime=regime,
                    underwater_state=underwater_state,
                    allocation=allocation,
                    rsi_buy_allowed=_rsi_buy_allowed(indicators),
                    intent_count=len(recovery_intents),
                ),
                "de_risk": de_risk_diagnostics(
                    regime=regime,
                    position_context=position_context,
                    rsi_sell_allowed=_rsi_sell_allowed(indicators),
                    intent_count=len(de_risk_intents),
                ),
            },
            regime=regime,
            intents=tuple(intents),
        )


def _quantize_price(value: Decimal) -> Decimal:
    return value.quantize(DEFAULT_PRICE_QUANT, rounding=ROUND_HALF_UP)


def _build_uptrend_grid(
    *,
    symbol: str,
    timeframe: str,
    reference_price: Decimal,
    range_low: Decimal,
    range_high: Decimal,
    max_grid_levels: int,
    max_position_fraction: Decimal,
    min_price_distance_fraction: Decimal,
    allocation: PortfolioAllocationSnapshot | None,
    entry_allowed: bool,
    indicators: IndicatorSnapshot | None,
    market_structure: MarketStructureSnapshot | None,
) -> tuple[tuple[GridLevel, ...], tuple[TargetIntent, ...]]:
    if not entry_allowed or not _rsi_buy_allowed(indicators):
        return (), ()
    if reference_price <= Decimal("0"):
        raise ValueError("reference_price must be positive")

    atr_step = _uptrend_atr_step(reference_price=reference_price, indicators=indicators)
    support_prices = _support_prices_below_reference(market_structure, reference_price=reference_price)
    resistance_price = _nearest_resistance_above_reference(market_structure, reference_price=reference_price)
    price_band = PriceBand(
        range_low=_quantize_price(
            market_structure.range_low
            if market_structure is not None and market_structure.range_low is not None
            else range_low
        ),
        range_high=_quantize_price(resistance_price or range_high),
    )
    risk = _entry_risk(max_position_fraction=max_position_fraction, allocation=allocation)
    levels: list[GridLevel] = []
    intents: list[TargetIntent] = []
    used_prices: set[Decimal] = set()

    for index in range(max_grid_levels):
        raw_pullback = _quantize_price(reference_price - (Decimal(index + 1) * atr_step))
        target_price, alignment_reason = _align_pullback_to_support(
            raw_pullback,
            support_prices=support_prices,
            atr_step=atr_step,
            used_prices=used_prices,
        )
        if (
            target_price <= Decimal("0")
            or target_price >= reference_price
            or target_price in used_prices
            or not _price_distance_allowed(
                target_price,
                reference_price=reference_price,
                min_price_distance_fraction=min_price_distance_fraction,
            )
        ):
            continue
        used_prices.add(target_price)
        reason_codes = ("uptrend_pullback_buy", alignment_reason, "atr_step_spacing", "rsi_buy_passed")
        levels.append(
            GridLevel(
                side=GridLevelSide.BUY,
                price=target_price,
                level_index=index,
                reason=reason_codes[0],
            )
        )
        intents.append(
            TargetIntent(
                intent_type=TargetIntentType.OPEN_POSITION,
                execution_intent=TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE,
                symbol=symbol,
                timeframe=timeframe,
                regime=MarketRegime.UPTREND,
                side=GridLevelSide.BUY,
                target_price=target_price,
                reference_price=reference_price,
                price_band=price_band,
                risk=risk,
                guards=StrategyGuardSnapshot(
                    buy_allowed=True,
                    sell_allowed=False,
                    rsi14=indicators.rsi14 if indicators is not None else None,
                    atr14=indicators.atr14 if indicators is not None else None,
                ),
                reason_codes=reason_codes,
                grid_level_index=index,
                metadata={
                    "raw_pullback_price": raw_pullback,
                    "atr_step": atr_step,
                },
            )
        )

    return tuple(levels), tuple(intents)


def _uptrend_atr_step(*, reference_price: Decimal, indicators: IndicatorSnapshot | None) -> Decimal:
    if indicators is not None and indicators.atr14 is not None and indicators.atr14 > Decimal("0"):
        return _quantize_price(indicators.atr14)
    return _quantize_price(max(reference_price * Decimal("0.01"), DEFAULT_PRICE_QUANT))


def _support_prices_below_reference(
    market_structure: MarketStructureSnapshot | None,
    *,
    reference_price: Decimal,
) -> tuple[Decimal, ...]:
    if market_structure is None:
        return ()
    return tuple(
        candidate.price
        for candidate in market_structure.support_candidates
        if Decimal("0") < candidate.price < reference_price
    )


def _nearest_resistance_above_reference(
    market_structure: MarketStructureSnapshot | None,
    *,
    reference_price: Decimal,
) -> Decimal | None:
    if market_structure is None:
        return None
    candidates = tuple(
        candidate.price
        for candidate in market_structure.resistance_candidates
        if candidate.price > reference_price
    )
    return min(candidates) if candidates else None


def _align_pullback_to_support(
    raw_pullback: Decimal,
    *,
    support_prices: tuple[Decimal, ...],
    atr_step: Decimal,
    used_prices: set[Decimal],
) -> tuple[Decimal, str]:
    candidates = tuple(price for price in support_prices if price not in used_prices)
    if not candidates:
        return raw_pullback, "atr_pullback"
    nearest = min(candidates, key=lambda price: abs(price - raw_pullback))
    if abs(nearest - raw_pullback) <= atr_step:
        return _quantize_price(nearest), "support_aligned"
    return raw_pullback, "atr_pullback"


def _build_range_grid(
    *,
    symbol: str,
    timeframe: str,
    reference_price: Decimal,
    range_low: Decimal,
    range_high: Decimal,
    max_grid_levels: int,
    max_position_fraction: Decimal,
    min_price_distance_fraction: Decimal,
    allocation: PortfolioAllocationSnapshot | None,
    regime: MarketRegime,
    entry_block_reasons: tuple[str, ...],
    indicators: IndicatorSnapshot | None,
    position_context: PositionContext | None,
) -> tuple[tuple[GridLevel, ...], tuple[TargetIntent, ...]]:
    if reference_price <= Decimal("0"):
        raise ValueError("reference_price must be positive")
    if range_low <= Decimal("0"):
        raise ValueError("range_low must be positive")
    if not range_low < reference_price < range_high:
        raise ValueError("reference_price must be inside range band")

    levels: list[GridLevel] = []
    intents: list[TargetIntent] = []
    entry_allowed = (
        regime not in (MarketRegime.DOWNTREND, MarketRegime.RISK_OFF)
        and not entry_block_reasons
        and _rsi_buy_allowed(indicators)
    )
    sell_allowed = _rsi_sell_allowed(indicators)
    min_no_loss_exit_price = _min_no_loss_exit_price(position_context)
    buy_step = (reference_price - range_low) / Decimal(max_grid_levels + 1)
    sell_step = (range_high - reference_price) / Decimal(max_grid_levels + 1)
    price_band = PriceBand(range_low=range_low, range_high=range_high)
    entry_risk = _entry_risk(max_position_fraction=max_position_fraction, allocation=allocation)
    exit_risk = StrategyRiskLimits(max_position_fraction=max_position_fraction)

    for index in range(max_grid_levels):
        if entry_allowed:
            buy_price = _quantize_price(reference_price - (Decimal(index + 1) * buy_step))
            if (
                range_low < buy_price < reference_price
                and _price_distance_allowed(
                    buy_price,
                    reference_price=reference_price,
                    min_price_distance_fraction=min_price_distance_fraction,
                )
            ):
                levels.append(
                    GridLevel(
                        side=GridLevelSide.BUY,
                        price=buy_price,
                        level_index=index,
                        reason="range_buy",
                    )
                )
                intents.append(
                    TargetIntent(
                        intent_type=TargetIntentType.OPEN_POSITION,
                        execution_intent=TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE,
                        symbol=symbol,
                        timeframe=timeframe,
                        regime=regime,
                        side=GridLevelSide.BUY,
                        target_price=buy_price,
                        reference_price=reference_price,
                        price_band=price_band,
                        risk=entry_risk,
                        guards=StrategyGuardSnapshot(
                            buy_allowed=True,
                            sell_allowed=False,
                            rsi14=indicators.rsi14 if indicators is not None else None,
                            atr14=indicators.atr14 if indicators is not None else None,
                        ),
                        reason_codes=("range_buy", "rsi_buy_passed"),
                        grid_level_index=index,
                    )
                )

        sell_price = _quantize_price(reference_price + (Decimal(index + 1) * sell_step))
        no_loss_passed = min_no_loss_exit_price is not None and sell_price >= min_no_loss_exit_price
        if (
            sell_allowed
            and no_loss_passed
            and reference_price < sell_price < range_high
            and _price_distance_allowed(
                sell_price,
                reference_price=reference_price,
                min_price_distance_fraction=min_price_distance_fraction,
            )
        ):
            sell_position = _position_with_min_no_loss_exit_price(position_context, min_no_loss_exit_price)
            levels.append(
                GridLevel(
                    side=GridLevelSide.SELL,
                    price=sell_price,
                    level_index=index,
                    reason="range_take_profit",
                )
            )
            intents.append(
                TargetIntent(
                    intent_type=TargetIntentType.CLOSE_POSITION,
                    execution_intent=TargetExecutionIntent.LIMIT_EXIT_CANDIDATE,
                    symbol=symbol,
                    timeframe=timeframe,
                    regime=regime,
                    side=GridLevelSide.SELL,
                    target_price=sell_price,
                    reference_price=reference_price,
                    price_band=price_band,
                    risk=exit_risk,
                    guards=StrategyGuardSnapshot(
                        buy_allowed=False,
                        sell_allowed=True,
                        rsi14=indicators.rsi14 if indicators is not None else None,
                        atr14=indicators.atr14 if indicators is not None else None,
                        no_loss_required=True,
                        no_loss_passed=True,
                    ),
                    reason_codes=("range_take_profit", "rsi_sell_passed", "no_loss_passed"),
                    grid_level_index=index,
                    position=sell_position,
                )
            )

    return tuple(levels), tuple(intents)


def _rsi_buy_allowed(indicators: IndicatorSnapshot | None) -> bool:
    return indicators is not None and indicators.rsi14 is not None and indicators.rsi14 <= RSI_BUY_THRESHOLD


def _rsi_sell_allowed(indicators: IndicatorSnapshot | None) -> bool:
    return indicators is not None and indicators.rsi14 is not None and indicators.rsi14 > RSI_SELL_THRESHOLD


def _rsi_block_reasons(indicators: IndicatorSnapshot | None) -> tuple[str, ...]:
    if indicators is None or indicators.rsi14 is None:
        return ("rsi_unavailable_buy_block", "rsi_unavailable_sell_block")
    reasons: list[str] = []
    if not _rsi_buy_allowed(indicators):
        reasons.append("rsi_buy_block")
    if not _rsi_sell_allowed(indicators):
        reasons.append("rsi_sell_block")
    return tuple(reasons)


def _allocation_for_symbol(
    *,
    portfolio_context: PortfolioContext | None,
    symbol: str,
    max_position_fraction: Decimal,
    candidate_count: int,
) -> PortfolioAllocationSnapshot | None:
    if portfolio_context is None:
        return None
    return allocate_portfolio_budget(
        portfolio_context=portfolio_context,
        symbol=symbol,
        max_position_fraction=max_position_fraction,
        candidate_count=candidate_count,
    )


def _entry_risk(
    *,
    max_position_fraction: Decimal,
    allocation: PortfolioAllocationSnapshot | None,
) -> StrategyRiskLimits:
    if allocation is None:
        return StrategyRiskLimits(max_position_fraction=max_position_fraction)
    return StrategyRiskLimits(
        max_position_fraction=max_position_fraction,
        suggested_quote_notional=allocation.suggested_quote_notional,
        max_quote_notional=allocation.max_quote_notional,
    )


def _empty_allocation_diagnostics(
    *,
    symbol: str,
    max_position_fraction: Decimal,
    candidate_count: int,
) -> dict[str, object]:
    return {
        "symbol": symbol.upper(),
        "max_position_fraction": str(max_position_fraction),
        "symbol_current_quote_notional": "0",
        "symbol_max_quote_notional": "0",
        "symbol_remaining_quote_notional": "0",
        "portfolio_available_quote": "0",
        "portfolio_remaining_quote_notional": "0",
        "max_quote_notional": "0",
        "suggested_quote_notional": "0",
        "candidate_count": candidate_count,
    }


def _price_distance_allowed(
    target_price: Decimal,
    *,
    reference_price: Decimal,
    min_price_distance_fraction: Decimal,
) -> bool:
    if min_price_distance_fraction <= Decimal("0"):
        return True
    if reference_price <= Decimal("0"):
        return False
    distance_fraction = abs(target_price - reference_price) / reference_price
    return distance_fraction >= min_price_distance_fraction


def _price_distance_diagnostics(
    *,
    reference_price: Decimal,
    intents: tuple[TargetIntent, ...],
    min_price_distance_fraction: Decimal,
) -> dict[str, object]:
    intent_distances = tuple(
        abs(intent.target_price - reference_price) / reference_price
        for intent in intents
        if intent.target_price is not None and reference_price > Decimal("0")
    )
    nearest_intent_distance = min(intent_distances) if intent_distances else None
    return {
        "min_price_distance_fraction": str(min_price_distance_fraction),
        "nearest_intent_distance_fraction": str(_quantize_price(nearest_intent_distance))
        if nearest_intent_distance is not None
        else None,
    }


def _min_no_loss_exit_price(position_context: PositionContext | None) -> Decimal | None:
    if position_context is None:
        return None
    if position_context.min_no_loss_exit_price is not None:
        return _quantize_price(position_context.min_no_loss_exit_price)
    if position_context.cost_basis is None:
        return None
    return _quantize_price(position_context.cost_basis * (Decimal("1") + NO_LOSS_EXIT_MARKUP))


def _position_for_symbol(
    portfolio_context: PortfolioContext | None,
    *,
    symbol: str,
) -> PositionContext | None:
    if portfolio_context is None:
        return None
    normalized_symbol = symbol.upper()
    return next((position for position in portfolio_context.positions if position.symbol == normalized_symbol), None)


def _no_loss_diagnostics(position_context: PositionContext | None) -> dict[str, object]:
    min_exit_price = _min_no_loss_exit_price(position_context)
    if position_context is None:
        return {
            "required": True,
            "passed": False,
            "cost_basis": None,
            "min_exit_price": None,
            "block_reason": "position_context_missing",
        }
    if position_context.cost_basis is None and position_context.min_no_loss_exit_price is None:
        return {
            "required": True,
            "passed": False,
            "cost_basis": None,
            "min_exit_price": None,
            "block_reason": "cost_basis_unknown",
        }
    return {
        "required": True,
        "passed": min_exit_price is not None,
        "cost_basis": str(position_context.cost_basis) if position_context.cost_basis is not None else None,
        "min_exit_price": str(min_exit_price) if min_exit_price is not None else None,
        "block_reason": None if min_exit_price is not None else "cost_basis_unknown",
    }


def _position_with_min_no_loss_exit_price(
    position_context: PositionContext | None,
    min_no_loss_exit_price: Decimal | None,
) -> PositionContext | None:
    if position_context is None:
        return None
    if min_no_loss_exit_price is None:
        return position_context
    return PositionContext(
        symbol=position_context.symbol,
        base_quantity=position_context.base_quantity,
        quote_notional=position_context.quote_notional,
        cost_basis=position_context.cost_basis,
        min_no_loss_exit_price=min_no_loss_exit_price,
    )
