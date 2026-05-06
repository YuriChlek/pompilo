from __future__ import annotations

from domain.market_structure import StructureBias, StructureSnapshot, detect_market_structure
from domain.models import Candle, IndicatorSnapshot, RegimeSnapshot, RegimeType
from domain.strategy_config import StrategyConfig


class MarketRegimeDetector:
    """Classify market regime by combining indicators with swing-based price structure."""

    def __init__(self, config: StrategyConfig) -> None:
        """Store strategy configuration for regime classification."""
        self.config = config

    def detect(
        self,
        candles: list[Candle],
        indicators: IndicatorSnapshot,
        risk_off: bool = False,
    ) -> RegimeSnapshot:
        """Return the current regime snapshot for one symbol candle history."""
        regime_snapshot, _ = self.detect_with_structure(candles, indicators, risk_off=risk_off)
        return regime_snapshot

    def extract_structure(self, candles: list[Candle]) -> StructureSnapshot:
        """Return the swing-based structure snapshot used by regime and grid planning."""
        return detect_market_structure(
            candles,
            swing_window=self.config.regime.structure_swing_window,
            lookback=self.config.regime.structure_lookback,
        )

    def detect_with_structure(
        self,
        candles: list[Candle],
        indicators: IndicatorSnapshot,
        risk_off: bool = False,
    ) -> tuple[RegimeSnapshot, StructureSnapshot]:
        """Return the current regime snapshot together with the extracted swing structure."""
        if risk_off:
            return RegimeSnapshot(RegimeType.RISK_OFF, 1.0, ["risk_manager"]), StructureSnapshot(
                bias=StructureBias.NEUTRAL,
                confidence=0.0,
                reasons=["risk_manager"],
            )

        latest_price = candles[-1].close
        atr_units_width = indicators.range_width / max(indicators.atr14, 1e-9)
        structure = self.extract_structure(candles)
        reasons: list[str] = []
        volume_confirmed = self._is_volume_confirmed(indicators)

        if indicators.atr_spike or indicators.abnormal_candle:
            reasons.append("volatility_spike")
            return RegimeSnapshot(RegimeType.HIGH_VOLATILITY, 0.95, reasons), structure

        bullish_ema = (
            indicators.ema20 > indicators.ema50 > indicators.ema200
            and indicators.ema50_slope >= self.config.regime.ema_mid_slope_trend_threshold
            and latest_price > indicators.ema50
        )
        bearish_ema = (
            indicators.ema20 < indicators.ema50 < indicators.ema200
            and indicators.ema50_slope <= -self.config.regime.ema_mid_slope_trend_threshold
            and latest_price < indicators.ema50
        )
        flat_range = (
            abs(indicators.ema50_slope) <= self.config.regime.ema_mid_slope_flat_threshold
            and self.config.regime.range_width_atr_min <= atr_units_width <= self.config.regime.range_width_atr_max
            and indicators.directional_move <= self.config.regime.range_directional_threshold
        )
        soft_range = (
            abs(indicators.ema50_slope) <= self.config.regime.ema_mid_slope_flat_threshold * 1.5
            and atr_units_width <= self.config.regime.range_width_atr_max
            and indicators.directional_move <= self.config.regime.range_directional_threshold
        )

        if bullish_ema and structure.bias == StructureBias.BULLISH:
            reasons.extend(["ema_stack_bullish", "positive_slope", "price_above_ema50"])
            reasons.extend(structure.reasons)
            return RegimeSnapshot(
                RegimeType.UPTREND,
                min(0.95, 0.7 + structure.confidence * 0.25),
                self._with_volume_reason(reasons, volume_confirmed),
                volume_confirmed=volume_confirmed,
            ), structure

        if bearish_ema and structure.bias == StructureBias.BEARISH:
            reasons.extend(["ema_stack_bearish", "negative_slope", "price_below_ema50"])
            reasons.extend(structure.reasons)
            return RegimeSnapshot(
                RegimeType.DOWNTREND,
                min(0.95, 0.7 + structure.confidence * 0.25),
                self._with_volume_reason(reasons, volume_confirmed),
                volume_confirmed=volume_confirmed,
            ), structure

        if flat_range and structure.bias in {StructureBias.RANGE, StructureBias.MIXED, StructureBias.NEUTRAL}:
            reasons.extend(["flat_slope", "bounded_range", "low_directionality"])
            reasons.extend(structure.reasons)
            return RegimeSnapshot(RegimeType.RANGE, 0.8, reasons), structure

        if soft_range and structure.bias in {StructureBias.RANGE, StructureBias.MIXED, StructureBias.NEUTRAL}:
            reasons.extend(["soft_flat_slope", "contained_range", "low_directionality"])
            reasons.extend(structure.reasons)
            return RegimeSnapshot(RegimeType.RANGE, 0.72, reasons), structure

        if bullish_ema and structure.bias in {StructureBias.MIXED, StructureBias.NEUTRAL, StructureBias.RANGE}:
            reasons.extend(["bullish_ema_without_clean_structure", *structure.reasons])
            return RegimeSnapshot(RegimeType.RANGE, 0.6, reasons), structure

        if bearish_ema and structure.bias in {StructureBias.MIXED, StructureBias.NEUTRAL, StructureBias.RANGE}:
            reasons.extend(["bearish_ema_without_clean_structure", *structure.reasons])
            return RegimeSnapshot(RegimeType.RANGE, 0.6, reasons), structure

        if structure.bias == StructureBias.BEARISH and bullish_ema:
            reasons.extend(["bullish_ema_but_bearish_structure", *structure.reasons])
            return RegimeSnapshot(RegimeType.RANGE, 0.58, reasons), structure

        if structure.bias == StructureBias.BULLISH and bearish_ema:
            reasons.extend(["bearish_ema_but_bullish_structure", *structure.reasons])
            return RegimeSnapshot(RegimeType.RANGE, 0.58, reasons), structure

        if structure.bias == StructureBias.BEARISH and abs(indicators.ema50_slope) <= self.config.regime.ema_mid_slope_trend_threshold:
            reasons.extend(["bearish_structure_transition", *structure.reasons])
            return RegimeSnapshot(RegimeType.RANGE, 0.57, reasons), structure

        if structure.bias == StructureBias.BULLISH and abs(indicators.ema50_slope) <= self.config.regime.ema_mid_slope_trend_threshold:
            reasons.extend(["bullish_structure_transition", *structure.reasons])
            return RegimeSnapshot(RegimeType.RANGE, 0.57, reasons), structure

        if structure.bias == StructureBias.BULLISH and latest_price >= indicators.ema50:
            reasons.extend(["structure_bullish_bias", *structure.reasons])
            return RegimeSnapshot(
                RegimeType.UPTREND,
                0.65,
                self._with_volume_reason(reasons, volume_confirmed),
                volume_confirmed=volume_confirmed,
            ), structure

        if structure.bias == StructureBias.BEARISH and latest_price <= indicators.ema50:
            reasons.extend(["structure_bearish_bias", *structure.reasons])
            return RegimeSnapshot(
                RegimeType.DOWNTREND,
                0.65,
                self._with_volume_reason(reasons, volume_confirmed),
                volume_confirmed=volume_confirmed,
            ), structure

        if structure.bias in {StructureBias.RANGE, StructureBias.MIXED}:
            reasons.extend(["structure_range_bias", *structure.reasons])
            return RegimeSnapshot(RegimeType.RANGE, 0.55, reasons), structure

        fallback = RegimeType.UPTREND if latest_price >= indicators.ema50 else RegimeType.DOWNTREND
        reasons.extend(["fallback_bias", *structure.reasons])
        return RegimeSnapshot(
            fallback,
            0.55,
            self._with_volume_reason(reasons, volume_confirmed),
            volume_confirmed=volume_confirmed,
        ), structure

    def _is_volume_confirmed(self, indicators: IndicatorSnapshot) -> bool:
        """Return whether the latest candle volume confirms a breakout-like transition."""
        if not self.config.regime.volume_confirmation_enabled:
            return True
        baseline = indicators.volume_ma20
        if baseline <= 0:
            return True
        return indicators.current_volume >= baseline * self.config.regime.volume_confirmation_multiplier

    @staticmethod
    def _with_volume_reason(reasons: list[str], volume_confirmed: bool) -> list[str]:
        """Attach a concise volume reason to trend-like regime outputs."""
        return [*reasons, "volume_confirmed" if volume_confirmed else "volume_not_confirmed"]
