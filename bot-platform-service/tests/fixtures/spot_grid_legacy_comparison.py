from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterator

from bot_platform_service.trading_bots.spot_grid.domain import (
    IndicatorCandle,
    IndicatorInput,
    MarketRegime,
    PortfolioContext,
    PositionContext,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
    compute_core_indicators,
    compute_market_structure,
    detect_single_timeframe_regime,
)
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = REPO_ROOT / "spot_grid_bot"
LEGACY_EXPECTED_DEVIATIONS = {
    "indicator.realized_volatility": "legacy uses log returns; platform fixture runtime uses simple returns",
    "target_prices": "legacy emits exchange target order geometry; platform emits execution-neutral target intents",
    "reason_codes": "legacy level tags and platform intent reason codes use different vocabularies",
    "platform_context": "platform planner consumes signal-only portfolio/position context; legacy comparison side omits live runtime side effects",
}


@dataclass(frozen=True, slots=True)
class SpotGridLegacyComparisonScenario:
    name: str
    candles: tuple["ComparisonCandle", ...]
    expected_regime: str
    position_context: PositionContext | None = None
    portfolio_context: PortfolioContext | None = None


@dataclass(frozen=True, slots=True)
class ComparisonCandle:
    timestamp: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class PlannerComparisonSide:
    regime: str
    indicators: dict[str, Decimal | str | None]
    target_prices: tuple[Decimal, ...]
    reason_codes: tuple[str, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class PlannerComparisonResult:
    scenario_name: str
    legacy: PlannerComparisonSide
    platform: PlannerComparisonSide
    deviations: dict[str, str]

    @property
    def passed(self) -> bool:
        return self.legacy.regime == self.platform.regime and _deviations_are_documented(self.deviations)


class SpotGridLegacyComparisonHarness:
    """Test-only fixture runner comparing legacy and platform Spot Grid planning outputs."""

    def __init__(self, *, symbol: str = "ETHUSDT", timeframe: str = "1h") -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._indicator_runtime = FakeStockIndicatorsRuntime()

    def compare(self, scenario: SpotGridLegacyComparisonScenario) -> PlannerComparisonResult:
        legacy = self._run_legacy(scenario)
        platform = self._run_platform(scenario)
        deviations = _documented_deviations(legacy=legacy, platform=platform, scenario=scenario)
        return PlannerComparisonResult(
            scenario_name=scenario.name,
            legacy=legacy,
            platform=platform,
            deviations=deviations,
        )

    def _run_legacy(self, scenario: SpotGridLegacyComparisonScenario) -> PlannerComparisonSide:
        with _legacy_import_path():
            from domain.grid_builder import GridBuilder
            from domain.indicators import compute_snapshot
            from domain.models import RegimeType
            from domain.regime_detector import MarketRegimeDetector
            from domain.strategy_config import DEFAULT_STRATEGY_CONFIG

            legacy_candles = [_legacy_candle(candle) for candle in scenario.candles]
            indicators = compute_snapshot(legacy_candles, DEFAULT_STRATEGY_CONFIG)
            regime, structure = MarketRegimeDetector(DEFAULT_STRATEGY_CONFIG).detect_with_structure(
                legacy_candles,
                indicators,
            )
            builder = GridBuilder(DEFAULT_STRATEGY_CONFIG)
            last_price = legacy_candles[-1].close
            if regime.regime is RegimeType.UPTREND:
                grid = builder.build_trend_pullback_grid(last_price, indicators)
            elif regime.regime is RegimeType.RANGE:
                grid = builder.build_range_grid(last_price, indicators, structure_snapshot=structure)
            else:
                grid = None
            return PlannerComparisonSide(
                regime=_legacy_regime_value(regime.regime),
                indicators={
                    "ema20": _float_decimal(indicators.ema20),
                    "ema50": _float_decimal(indicators.ema50),
                    "ema200": _float_decimal(indicators.ema200),
                    "atr14": _float_decimal(indicators.atr14),
                    "rsi14": _float_decimal(indicators.rsi14),
                    "realized_volatility": _float_decimal(indicators.realized_volatility),
                },
                target_prices=tuple(_float_decimal(level.price) for level in (grid.levels if grid else ())),
                reason_codes=tuple(level.tag for level in (grid.levels if grid else ())),
                diagnostics={},
            )

    def _run_platform(self, scenario: SpotGridLegacyComparisonScenario) -> PlannerComparisonSide:
        indicator_input = IndicatorInput(
            source="binance_spot",
            symbol=self.symbol,
            timeframe=self.timeframe,
            snapshot_id=f"fixture-{scenario.name}",
            snapshot_version=1,
            data_hash=f"hash-{scenario.name}",
            candles=tuple(_platform_indicator_candle(candle) for candle in scenario.candles),
            required_history=200,
        )
        indicators = compute_core_indicators(indicator_input, runtime=self._indicator_runtime)
        market_structure = compute_market_structure(indicator_input.candles)
        regime = detect_single_timeframe_regime(indicators=indicators, market_structure=market_structure)
        plan = SpotGridPlanner().plan(
            symbol=self.symbol,
            timeframe=self.timeframe,
            candles=tuple(_platform_planner_candle(candle) for candle in scenario.candles),
            config=SpotGridConfig(
                symbols=(self.symbol,),
                primary_timeframe=self.timeframe,
                supporting_timeframes=(),
                max_position_fraction=Decimal("0.10"),
                max_grid_levels=3,
            ),
            regime=regime.regime,
            indicators=indicators,
            market_structure=market_structure,
            portfolio_context=scenario.portfolio_context,
            position_context=scenario.position_context,
        )
        return PlannerComparisonSide(
            regime=regime.regime.value,
            indicators={
                "ema20": indicators.ema20,
                "ema50": indicators.ema50,
                "ema200": indicators.ema200,
                "atr14": indicators.atr14,
                "rsi14": indicators.rsi14,
                "realized_volatility": indicators.realized_volatility,
            },
            target_prices=tuple(intent.target_price for intent in plan.intents if intent.target_price is not None),
            reason_codes=tuple(reason for intent in plan.intents for reason in intent.reason_codes),
            diagnostics=plan.diagnostics,
        )


def basic_legacy_comparison_scenarios() -> tuple[SpotGridLegacyComparisonScenario, ...]:
    return (
        SpotGridLegacyComparisonScenario(
            name="range",
            candles=_range_candles(),
            expected_regime=MarketRegime.RANGE.value,
        ),
        SpotGridLegacyComparisonScenario(
            name="uptrend",
            candles=_trend_candles(direction=Decimal("1")),
            expected_regime=MarketRegime.UPTREND.value,
        ),
        SpotGridLegacyComparisonScenario(
            name="downtrend",
            candles=_trend_candles(direction=Decimal("-1")),
            expected_regime=MarketRegime.DOWNTREND.value,
        ),
    )


def extended_legacy_comparison_scenarios() -> tuple[SpotGridLegacyComparisonScenario, ...]:
    return (
        *basic_legacy_comparison_scenarios(),
        SpotGridLegacyComparisonScenario(
            name="high_volatility_range_spike",
            candles=_high_volatility_candles(direction=Decimal("0")),
            expected_regime=MarketRegime.HIGH_VOLATILITY.value,
        ),
        SpotGridLegacyComparisonScenario(
            name="high_volatility_upward_spike",
            candles=_high_volatility_candles(direction=Decimal("0.04")),
            expected_regime=MarketRegime.HIGH_VOLATILITY.value,
        ),
        SpotGridLegacyComparisonScenario(
            name="underwater_range_recovery",
            candles=_range_candles(),
            expected_regime=MarketRegime.RANGE.value,
            portfolio_context=_portfolio_context(quote_notional=Decimal("120"), cost_basis=Decimal("120")),
        ),
        SpotGridLegacyComparisonScenario(
            name="underwater_range_budget_block",
            candles=_range_candles(),
            expected_regime=MarketRegime.RANGE.value,
            portfolio_context=_portfolio_context(quote_notional=Decimal("200"), cost_basis=Decimal("120")),
        ),
        SpotGridLegacyComparisonScenario(
            name="no_loss_explicit_threshold_block",
            candles=_range_candles(),
            expected_regime=MarketRegime.RANGE.value,
            position_context=PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("1"),
                quote_notional=Decimal("100"),
                cost_basis=Decimal("109"),
                min_no_loss_exit_price=Decimal("109.50"),
            ),
        ),
        SpotGridLegacyComparisonScenario(
            name="no_loss_unknown_cost_basis_block",
            candles=_range_candles(),
            expected_regime=MarketRegime.RANGE.value,
            position_context=PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("1"),
                quote_notional=Decimal("100"),
                cost_basis=None,
                min_no_loss_exit_price=None,
            ),
        ),
        SpotGridLegacyComparisonScenario(
            name="range_sell_no_loss_pass",
            candles=_range_candles(),
            expected_regime=MarketRegime.RANGE.value,
            position_context=PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("1"),
                quote_notional=Decimal("100"),
                cost_basis=Decimal("100"),
                min_no_loss_exit_price=None,
            ),
        ),
    )


def _documented_deviations(
    *,
    legacy: PlannerComparisonSide,
    platform: PlannerComparisonSide,
    scenario: SpotGridLegacyComparisonScenario,
) -> dict[str, str]:
    deviations: dict[str, str] = {}
    if legacy.target_prices != platform.target_prices:
        deviations["target_prices"] = LEGACY_EXPECTED_DEVIATIONS["target_prices"]
    if legacy.reason_codes != platform.reason_codes:
        deviations["reason_codes"] = LEGACY_EXPECTED_DEVIATIONS["reason_codes"]
    if _indicator_delta(legacy, platform, "realized_volatility") > Decimal("0.00000001"):
        deviations["indicator.realized_volatility"] = LEGACY_EXPECTED_DEVIATIONS["indicator.realized_volatility"]
    if scenario.position_context is not None or scenario.portfolio_context is not None:
        deviations["platform_context"] = LEGACY_EXPECTED_DEVIATIONS["platform_context"]
    return deviations


def _deviations_are_documented(deviations: dict[str, str]) -> bool:
    return all(deviation in LEGACY_EXPECTED_DEVIATIONS for deviation in deviations)


def _indicator_delta(legacy: PlannerComparisonSide, platform: PlannerComparisonSide, name: str) -> Decimal:
    legacy_value = legacy.indicators[name]
    platform_value = platform.indicators[name]
    if not isinstance(legacy_value, Decimal) or not isinstance(platform_value, Decimal):
        return Decimal("0")
    return abs(legacy_value - platform_value)


@contextmanager
def _legacy_import_path() -> Iterator[None]:
    legacy_path = str(LEGACY_ROOT)
    inserted = False
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(legacy_path)


def _legacy_candle(candle: ComparisonCandle):
    from domain.models import Candle

    return Candle(
        timestamp=candle.timestamp,
        open=float(candle.open),
        high=float(candle.high),
        low=float(candle.low),
        close=float(candle.close),
        volume=float(candle.volume),
    )


def _platform_indicator_candle(candle: ComparisonCandle) -> IndicatorCandle:
    return IndicatorCandle(
        timestamp=f"2026-07-01T{candle.timestamp % 24:02d}:00:00+00:00",
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
    )


def _platform_planner_candle(candle: ComparisonCandle) -> SpotGridCandle:
    return SpotGridCandle(
        timestamp=f"2026-07-01T{candle.timestamp % 24:02d}:00:00+00:00",
        open=candle.open,
        high=candle.high,
        low=candle.low,
        close=candle.close,
        volume=candle.volume,
    )


def _range_candles() -> tuple[ComparisonCandle, ...]:
    candles = []
    for index in range(220):
        close = Decimal("100")
        candles.append(
            ComparisonCandle(
                timestamp=index,
                open=close,
                high=close + Decimal("2.0"),
                low=close - Decimal("2.0"),
                close=close,
                volume=Decimal("1000"),
            )
        )
    return tuple(candles)


def _trend_candles(*, direction: Decimal) -> tuple[ComparisonCandle, ...]:
    candles = []
    base = Decimal("80") if direction > 0 else Decimal("180")
    for index in range(220):
        close = base + (Decimal(index) * direction * Decimal("0.35"))
        candles.append(
            ComparisonCandle(
                timestamp=index,
                open=close - (direction * Decimal("0.1")),
                high=close + Decimal("1.5"),
                low=close - Decimal("1.5"),
                close=close,
                volume=Decimal("1500") if index == 219 else Decimal("1000"),
            )
        )
    return tuple(candles)


def _high_volatility_candles(*, direction: Decimal) -> tuple[ComparisonCandle, ...]:
    candles = []
    for index in range(220):
        baseline = Decimal("100") + (Decimal(index) * direction)
        swing = Decimal("10") if index % 2 == 0 else Decimal("-10")
        close = baseline + swing
        high = close + Decimal("2")
        low = close - Decimal("2")
        if index == 219:
            close = Decimal("100")
            high = Decimal("170")
            low = Decimal("30")
        candles.append(
            ComparisonCandle(
                timestamp=index,
                open=baseline,
                high=high,
                low=low,
                close=close,
                volume=Decimal("1000"),
            )
        )
    return tuple(candles)


def _portfolio_context(*, quote_notional: Decimal, cost_basis: Decimal) -> PortfolioContext:
    return PortfolioContext(
        total_equity=Decimal("1000"),
        available_quote=Decimal("300"),
        positions=(
            PositionContext(
                symbol="ETHUSDT",
                base_quantity=Decimal("1"),
                quote_notional=quote_notional,
                cost_basis=cost_basis,
            ),
        ),
    )


def _legacy_regime_value(regime: object) -> str:
    return str(getattr(regime, "value", regime)).lower()


def _float_decimal(value: float) -> Decimal:
    return Decimal(str(value))
