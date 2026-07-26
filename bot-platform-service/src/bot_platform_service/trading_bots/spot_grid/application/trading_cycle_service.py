from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Mapping

from bot_platform_service.domain import BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_grid.bot_config import parse_spot_grid_config
from bot_platform_service.trading_bots.spot_grid.domain import (
    IndicatorInput,
    IndicatorRuntime,
    IndicatorSnapshot,
    MarketRegime,
    PortfolioContext,
    RegimeSnapshot,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlan,
    SpotGridPlanner,
    compute_core_indicators,
    compute_exposure,
    compute_market_structure,
    detect_single_timeframe_regime,
    detect_underwater_state,
    resolve_regime_state,
)
from bot_platform_service.trading_bots.spot_grid.infrastructure import PlatformSnapshotIndicatorAdapter


@dataclass(frozen=True, slots=True)
class SpotGridCycleResult:
    """One platform-native Spot Grid planning result."""

    plan: SpotGridPlan
    diagnostics: dict[str, object]


class SpotGridTradingCycleService:
    """Application service for one side-effect-free Spot Grid planning cycle."""

    def __init__(
        self,
        planner: SpotGridPlanner | None = None,
        snapshot_adapter: PlatformSnapshotIndicatorAdapter | None = None,
        indicator_runtime: IndicatorRuntime | None = None,
    ) -> None:
        self.planner = planner or SpotGridPlanner()
        self.snapshot_adapter = snapshot_adapter or PlatformSnapshotIndicatorAdapter()
        self.indicator_runtime = indicator_runtime

    def run_once(
        self,
        *,
        market_data: BotMarketDataContext,
        config: SpotGridConfig,
        previous_regime_state: Mapping[str, object] | None = None,
        previous_cooldown_state: Mapping[str, object] | None = None,
        portfolio_context: PortfolioContext | None = None,
    ) -> SpotGridCycleResult:
        """Plan one Spot Grid cycle from platform primary and supporting snapshots."""
        primary_snapshot = market_data.primary_snapshot
        indicator_input = self.snapshot_adapter.adapt(primary_snapshot)
        indicator_snapshot = compute_core_indicators(indicator_input, runtime=self.indicator_runtime)
        market_structure = compute_market_structure(indicator_input.candles)
        regime_snapshot = detect_single_timeframe_regime(
            indicators=indicator_snapshot,
            market_structure=market_structure,
        )
        regime_state = resolve_regime_state(
            symbol=indicator_input.symbol,
            timeframe=indicator_input.timeframe,
            detected=regime_snapshot,
            snapshot_id=primary_snapshot.snapshot_id,
            snapshot_version=primary_snapshot.snapshot_version,
            data_hash=primary_snapshot.data_hash,
            previous_state=previous_regime_state,
        )
        confirmation = _multi_timeframe_confirmation(
            supporting_snapshots=market_data.supporting_snapshots,
            config=config,
            snapshot_adapter=self.snapshot_adapter,
            indicator_runtime=self.indicator_runtime,
        )
        volatility_pause = _volatility_pause(
            indicators=indicator_snapshot,
            config=config,
            previous_cooldown_state=previous_cooldown_state,
            snapshot_id=primary_snapshot.snapshot_id,
            snapshot_version=primary_snapshot.snapshot_version,
            data_hash=primary_snapshot.data_hash,
        )
        entry_block_reasons = (
            *confirmation["entry_block_reasons"],
            *volatility_pause["entry_block_reasons"],
        )
        resolved_portfolio_context = portfolio_context or _empty_portfolio_context()
        exposure = compute_exposure(resolved_portfolio_context)
        reference_price = indicator_input.candles[-1].close if indicator_input.candles else Decimal("0")
        underwater_state = detect_underwater_state(
            portfolio_context=resolved_portfolio_context,
            symbol=indicator_input.symbol,
            reference_price=reference_price,
            regime=regime_state.effective_regime,
            entry_block_reasons=entry_block_reasons,
        )
        candles = tuple(
            SpotGridCandle(
                timestamp=candle.timestamp,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
            for candle in indicator_input.candles
        )
        plan = self.planner.plan(
            symbol=indicator_input.symbol,
            timeframe=indicator_input.timeframe,
            candles=candles,
            config=config,
            regime=regime_state.effective_regime,
            entry_block_reasons=entry_block_reasons,
            indicators=indicator_snapshot,
            market_structure=market_structure,
            portfolio_context=resolved_portfolio_context,
        )
        supporting = tuple(
            {
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_version": snapshot.snapshot_version,
                "timeframe": snapshot.timeframe,
                "data_hash": snapshot.data_hash,
            }
            for snapshot in market_data.supporting_snapshots
        )
        return SpotGridCycleResult(
            plan=plan,
            diagnostics={
                **plan.diagnostics,
                "snapshot_id": primary_snapshot.snapshot_id,
                "snapshot_version": primary_snapshot.snapshot_version,
                "data_hash": primary_snapshot.data_hash,
                "configured_symbols": config.symbols,
                "subscribed_symbols": config.symbols,
                "primary_timeframe": primary_snapshot.timeframe,
                "indicator_candle_count": indicator_input.candle_count,
                "indicator_required_history": indicator_input.required_history,
                "indicator_has_required_history": indicator_input.has_required_history,
                "indicator_volatility_has_required_history": indicator_snapshot.volatility_has_required_history,
                "indicator_volume_has_required_history": indicator_snapshot.volume_has_required_history,
                "indicators": indicator_snapshot.to_payload(),
                "market_structure": market_structure.to_payload(),
                "regime": regime_state.effective_regime.value,
                "regime_decision": regime_snapshot.to_payload(),
                "regime_state": regime_state.to_payload(),
                "multi_timeframe_confirmation": confirmation["diagnostics"],
                "volatility_pause": volatility_pause["diagnostics"],
                "cooldown_state": volatility_pause["cooldown_state"],
                "portfolio_context": _portfolio_context_diagnostics(
                    portfolio_context=resolved_portfolio_context,
                    fallback_used=portfolio_context is None,
                ),
                "exposure": exposure.to_payload(),
                "underwater": underwater_state.to_payload(),
                "supporting_timeframes": tuple(snapshot["timeframe"] for snapshot in supporting),
                "supporting_snapshots": supporting,
            },
        )


def _multi_timeframe_confirmation(
    *,
    supporting_snapshots: tuple[BotMarketSnapshot, ...],
    config: SpotGridConfig,
    snapshot_adapter: PlatformSnapshotIndicatorAdapter,
    indicator_runtime: IndicatorRuntime | None,
) -> dict[str, object]:
    expected_4h = "4h" in config.supporting_timeframes
    supporting_4h = next((snapshot for snapshot in supporting_snapshots if snapshot.timeframe == "4h"), None)
    if supporting_4h is None:
        reason_codes = ("supporting_4h_missing",) if expected_4h else ()
        return {
            "entry_block_reasons": reason_codes,
            "diagnostics": {
                "required_timeframe": "4h",
                "status": "missing" if expected_4h else "not_configured",
                "new_buy_allowed": not reason_codes,
                "reason_codes": reason_codes,
                "supporting_regimes": (),
            },
        }

    indicator_input = snapshot_adapter.adapt(supporting_4h)
    supporting_regime = _detect_supporting_regime(indicator_input, indicator_runtime=indicator_runtime)
    reason_codes = _supporting_regime_block_reasons(supporting_regime.regime)
    return {
        "entry_block_reasons": reason_codes,
        "diagnostics": {
            "required_timeframe": "4h",
            "status": "confirmed",
            "new_buy_allowed": not reason_codes,
            "reason_codes": reason_codes,
            "supporting_regimes": (
                {
                    "snapshot_id": supporting_4h.snapshot_id,
                    "snapshot_version": supporting_4h.snapshot_version,
                    "timeframe": supporting_4h.timeframe,
                    "data_hash": supporting_4h.data_hash,
                    "regime": supporting_regime.regime.value,
                    "regime_decision": supporting_regime.to_payload(),
                },
            ),
        },
    }


def _detect_supporting_regime(
    indicator_input: IndicatorInput,
    *,
    indicator_runtime: IndicatorRuntime | None,
) -> RegimeSnapshot:
    indicators = compute_core_indicators(indicator_input, runtime=indicator_runtime)
    market_structure = compute_market_structure(indicator_input.candles)
    return detect_single_timeframe_regime(indicators=indicators, market_structure=market_structure)


def _supporting_regime_block_reasons(regime: MarketRegime) -> tuple[str, ...]:
    if regime is MarketRegime.DOWNTREND:
        return ("supporting_4h_downtrend_block",)
    if regime is MarketRegime.RISK_OFF:
        return ("supporting_4h_risk_off_block",)
    return ()


def _empty_portfolio_context() -> PortfolioContext:
    return PortfolioContext(total_equity=Decimal("0"), available_quote=Decimal("0"), positions=())


def _portfolio_context_diagnostics(
    *,
    portfolio_context: PortfolioContext,
    fallback_used: bool,
) -> dict[str, object]:
    return {
        "provided": not fallback_used,
        "fallback": "empty_conservative" if fallback_used else None,
        "total_equity": str(portfolio_context.total_equity),
        "available_quote": str(portfolio_context.available_quote),
        "position_count": len(portfolio_context.positions),
        "position_symbols": tuple(position.symbol for position in portfolio_context.positions),
    }


def _volatility_pause(
    *,
    indicators: IndicatorSnapshot,
    config: SpotGridConfig,
    previous_cooldown_state: Mapping[str, object] | None,
    snapshot_id: str,
    snapshot_version: int,
    data_hash: str,
) -> dict[str, object]:
    realized_volatility = indicators.realized_volatility
    high_volatility = (
        realized_volatility is not None
        and realized_volatility >= config.high_volatility_pause_threshold
    )
    previous_remaining = _previous_cooldown_remaining(previous_cooldown_state)
    if high_volatility:
        remaining_runs = config.volatility_cooldown_runs
        reason_codes = ("high_volatility_entry_pause",)
    elif previous_remaining > 0:
        remaining_runs = previous_remaining - 1
        reason_codes = ("volatility_cooldown_entry_pause",)
    else:
        remaining_runs = 0
        reason_codes = ()

    cooldown_state = {
        "snapshot_id": snapshot_id,
        "snapshot_version": snapshot_version,
        "data_hash": data_hash,
        "active": bool(high_volatility or previous_remaining > 0),
        "remaining_runs": remaining_runs,
        "high_volatility": high_volatility,
        "realized_volatility": str(realized_volatility) if realized_volatility is not None else None,
        "threshold": str(config.high_volatility_pause_threshold),
        "reason_codes": reason_codes,
    }
    return {
        "entry_block_reasons": reason_codes,
        "cooldown_state": cooldown_state,
        "diagnostics": {
            "active": cooldown_state["active"],
            "remaining_runs": remaining_runs,
            "previous_remaining_runs": previous_remaining,
            "high_volatility": high_volatility,
            "realized_volatility": cooldown_state["realized_volatility"],
            "threshold": cooldown_state["threshold"],
            "reason_codes": reason_codes,
        },
    }


def _previous_cooldown_remaining(previous_cooldown_state: Mapping[str, object] | None) -> int:
    if previous_cooldown_state is None:
        return 0
    raw_remaining = previous_cooldown_state.get("remaining_runs")
    try:
        return max(0, int(raw_remaining))
    except (TypeError, ValueError):
        return 0
