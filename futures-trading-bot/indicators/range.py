from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

from .models import FractalResult, RangeMarketResult


class RangeMarketDetector:
    """Detect range-market conditions from the latest fractals and ATR compression."""

    def __init__(
        self,
        cluster_threshold_pct: float = 1.5,
        atr_threshold_pct: float = 0.8,
        range_lookback: int = 20,
        boundary_tolerance_ratio: float = 0.15,
    ):
        """Store clustering and ATR thresholds used for range-market detection."""
        self.cluster_threshold_pct = cluster_threshold_pct
        self.atr_threshold_pct = atr_threshold_pct
        self.range_lookback = range_lookback
        self.boundary_tolerance_ratio = boundary_tolerance_ratio

    def detect_range_market(
        self,
        df: pd.DataFrame,
        fractal_analysis: FractalResult,
        atr_series: Optional[pd.Series] = None,
    ) -> RangeMarketResult:
        """Analyze recent fractal structure and ATR compression for range conditions."""
        result = RangeMarketResult.empty()

        cluster_analysis = self._analyze_last_fractals(fractal_analysis)
        result.fractal_pattern = cluster_analysis["pattern"]
        result.avg_upper = cluster_analysis["avg_upper"]
        result.avg_lower = cluster_analysis["avg_lower"]
        result.range_type = cluster_analysis["range_type"]
        result.cluster_strength = cluster_analysis["cluster_strength"]
        result.price_levels = cluster_analysis["price_levels"]
        result.upper_fractals_analyzed = cluster_analysis["upper_fractals_count"]
        result.lower_fractals_analyzed = cluster_analysis["lower_fractals_count"]

        atr_analysis = self._analyze_atr_for_range(df, cluster_analysis, atr_series)
        result.atr_pct = atr_analysis["atr_pct"]
        range_analysis = self._analyze_structured_range(df)
        result.price_levels.update(range_analysis["price_levels"])

        confidence_score = 0
        reasons: List[str] = []

        if cluster_analysis["upper_cluster_strength"] != "none":
            confidence_score += 40
            reasons.append(
                "Останні {count}/3 верхніх фрактали кластеризовані (сила: {strength})".format(
                    count=cluster_analysis["upper_cluster_count"],
                    strength=cluster_analysis["upper_cluster_strength"],
                )
            )

        if cluster_analysis["lower_cluster_strength"] != "none":
            confidence_score += 40
            reasons.append(
                "Останні {count}/3 нижніх фрактали кластеризовані (сила: {strength})".format(
                    count=cluster_analysis["lower_cluster_count"],
                    strength=cluster_analysis["lower_cluster_strength"],
                )
            )

        if atr_analysis["is_low_volatility"]:
            confidence_score += 20
            reasons.append(f"Низька волатильність (ATR: {atr_analysis['atr_pct']:.2f}%)")

        if range_analysis["upper_touches"] >= 2:
            confidence_score += 15
            reasons.append(
                f"Верхня межа діапазону тестувалась {range_analysis['upper_touches']} раз(и)"
            )

        if range_analysis["lower_touches"] >= 2:
            confidence_score += 15
            reasons.append(
                f"Нижня межа діапазону тестувалась {range_analysis['lower_touches']} раз(и)"
            )

        if range_analysis["is_directionally_balanced"]:
            confidence_score += 10
            reasons.append(
                "Рух усередині останнього вікна збалансований, без вираженого directional drift"
            )

        if (
            cluster_analysis["upper_cluster_strength"] != "none"
            and cluster_analysis["lower_cluster_strength"] != "none"
        ):
            confidence_score += 20
            reasons.append("Обидва типи останніх фракталів показують кластеризацію")

        result.reasons = reasons
        result.confidence = min(100, confidence_score)
        result.is_range = result.confidence >= 60
        return result

    def _analyze_last_fractals(self, fractal_analysis: FractalResult) -> Dict[str, Any]:
        """Build clustering statistics for the latest upper and lower fractals."""
        upper_fractals = fractal_analysis.last_3_upper_fractals
        lower_fractals = fractal_analysis.last_3_lower_fractals

        upper_cluster = self._analyze_small_cluster([f["price"] for f in upper_fractals])
        lower_cluster = self._analyze_small_cluster([f["price"] for f in lower_fractals])

        avg_upper = np.mean([f["price"] for f in upper_fractals]) if upper_fractals else 0
        avg_lower = np.mean([f["price"] for f in lower_fractals]) if lower_fractals else 0

        if upper_cluster["strength"] != "none" and lower_cluster["strength"] != "none":
            range_type = "both_cluster"
        elif upper_cluster["strength"] != "none":
            range_type = "upper_cluster"
        elif lower_cluster["strength"] != "none":
            range_type = "lower_cluster"
        else:
            range_type = "none"

        cluster_strength = self._determine_overall_cluster_strength(upper_cluster, lower_cluster)

        price_levels: Dict[str, float] = {}
        if upper_cluster["strength"] != "none":
            price_levels.update(
                {
                    "upper_cluster_price": upper_cluster["avg_price"],
                    "upper_cluster_high": upper_cluster["max_price"],
                    "upper_cluster_low": upper_cluster["min_price"],
                    "upper_cluster_range_pct": upper_cluster["range_pct"],
                }
            )

        if lower_cluster["strength"] != "none":
            price_levels.update(
                {
                    "lower_cluster_price": lower_cluster["avg_price"],
                    "lower_cluster_high": lower_cluster["max_price"],
                    "lower_cluster_low": lower_cluster["min_price"],
                    "lower_cluster_range_pct": lower_cluster["range_pct"],
                }
            )

        pattern = self._identify_last_fractals_pattern(upper_cluster, lower_cluster)

        return {
            "pattern": pattern,
            "avg_upper": avg_upper,
            "avg_lower": avg_lower,
            "range_type": range_type,
            "cluster_strength": cluster_strength,
            "price_levels": price_levels,
            "upper_cluster_strength": upper_cluster["strength"],
            "upper_cluster_count": upper_cluster["count"],
            "lower_cluster_strength": lower_cluster["strength"],
            "lower_cluster_count": lower_cluster["count"],
            "upper_fractals_count": len(upper_fractals),
            "lower_fractals_count": len(lower_fractals),
        }

    def _analyze_small_cluster(self, prices: List[float]) -> Dict[str, Any]:
        """Determine whether a small set of fractal prices forms a meaningful cluster."""
        if len(prices) < 2:
            return self._empty_cluster()

        if len(prices) == 3:
            sorted_prices = sorted(prices)
            clusters = []

            diff_pct_12 = abs(sorted_prices[1] - sorted_prices[0]) / sorted_prices[0] * 100
            if diff_pct_12 < self.cluster_threshold_pct:
                clusters.append({"prices": sorted_prices[:2], "count": 2, "diff_pct": diff_pct_12})

            diff_pct_23 = abs(sorted_prices[2] - sorted_prices[1]) / sorted_prices[1] * 100
            if diff_pct_23 < self.cluster_threshold_pct:
                clusters.append({"prices": sorted_prices[1:], "count": 2, "diff_pct": diff_pct_23})

            total_range_pct = (sorted_prices[2] - sorted_prices[0]) / sorted_prices[0] * 100
            if total_range_pct < self.cluster_threshold_pct * 1.5:
                clusters.append({"prices": sorted_prices, "count": 3, "diff_pct": total_range_pct})

            if clusters:
                best_cluster = max(clusters, key=lambda x: x["count"])
                cluster_prices = best_cluster["prices"]
                count = best_cluster["count"]
                range_pct = best_cluster["diff_pct"]
            else:
                return self._empty_cluster()
        elif len(prices) == 2:
            diff_pct = abs(prices[1] - prices[0]) / prices[0] * 100
            if diff_pct < self.cluster_threshold_pct:
                cluster_prices = sorted(prices)
                count = 2
                range_pct = diff_pct
            else:
                return self._empty_cluster()
        else:
            return self._empty_cluster()

        if count == 3:
            if range_pct < self.cluster_threshold_pct * 0.7:
                strength = "strong"
            elif range_pct < self.cluster_threshold_pct:
                strength = "medium"
            else:
                strength = "weak"
        elif count == 2:
            strength = "medium" if range_pct < self.cluster_threshold_pct * 0.5 else "weak"
        else:
            strength = "none"

        avg_price = float(np.mean(cluster_prices))
        return {
            "strength": strength,
            "count": count,
            "avg_price": avg_price,
            "max_price": max(cluster_prices),
            "min_price": min(cluster_prices),
            "range_pct": range_pct,
            "prices": cluster_prices,
        }

    def _empty_cluster(self) -> Dict[str, Any]:
        """Return an empty cluster payload used when clustering is not available."""
        return {
            "strength": "none",
            "count": 0,
            "avg_price": 0,
            "max_price": 0,
            "min_price": 0,
            "range_pct": 0,
            "prices": [],
        }

    def _determine_overall_cluster_strength(self, upper_cluster: Dict[str, Any], lower_cluster: Dict[str, Any]) -> str:
        """Combine upper and lower cluster strengths into one overall label."""
        strengths = []
        if upper_cluster["strength"] != "none":
            strengths.append(upper_cluster["strength"])
        if lower_cluster["strength"] != "none":
            strengths.append(lower_cluster["strength"])

        if not strengths:
            return "none"

        strength_map = {"none": 0, "weak": 1, "medium": 2, "strong": 3}
        return max(strengths, key=lambda x: strength_map[x])

    def _identify_last_fractals_pattern(
        self, upper_cluster: Dict[str, Any], lower_cluster: Dict[str, Any]
    ) -> str:
        """Classify the latest fractal structure into a range-pattern label."""
        if upper_cluster["strength"] != "none" and lower_cluster["strength"] != "none":
            return "clear_range"
        if upper_cluster["strength"] != "none":
            return "resistance_forming"
        if lower_cluster["strength"] != "none":
            return "support_forming"
        return "no_cluster"

    def _analyze_atr_for_range(
        self,
        df: pd.DataFrame,
        cluster_analysis: Dict[str, Any],
        atr_series: Optional[pd.Series],
    ) -> Dict[str, Any]:
        """Measure ATR compression and derive volatility context for range detection."""
        default_result = {
            "atr_pct": 0.0,
            "is_low_volatility": False,
            "threshold_used": self.atr_threshold_pct,
        }

        if len(df) == 0:
            return default_result

        if atr_series is None:
            if len(df) < 14:
                return default_result
            atr_values = ta.atr(df["high"], df["low"], df["close"], length=14)
            if atr_values is None or atr_values.empty:
                return default_result
            current_atr = atr_values.iloc[-1]
        else:
            if atr_series.empty:
                return default_result
            current_atr = atr_series.iloc[-1]

        if pd.isna(current_atr):
            return default_result

        current_price = df["close"].iloc[-1]
        atr_pct = (current_atr / current_price) * 100 if current_price else 0.0

        effective_threshold = self.atr_threshold_pct
        if cluster_analysis["cluster_strength"] == "strong":
            effective_threshold *= 0.6
        elif cluster_analysis["cluster_strength"] == "medium":
            effective_threshold *= 0.8

        is_low_volatility = atr_pct < effective_threshold
        return {"atr_pct": atr_pct, "is_low_volatility": is_low_volatility, "threshold_used": effective_threshold}

    def _analyze_structured_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract range bounds, touch counts, and current price position within the latest window."""
        if len(df) == 0:
            return {
                "upper_touches": 0,
                "lower_touches": 0,
                "is_directionally_balanced": False,
                "price_levels": {},
            }

        recent = df.tail(min(len(df), self.range_lookback))
        upper_bound = float(recent["high"].max())
        lower_bound = float(recent["low"].min())
        last_close = float(recent["close"].iloc[-1])
        first_close = float(recent["close"].iloc[0])

        if upper_bound <= lower_bound:
            return {
                "upper_touches": 0,
                "lower_touches": 0,
                "is_directionally_balanced": False,
                "price_levels": {
                    "range_upper_bound": upper_bound,
                    "range_lower_bound": lower_bound,
                    "range_width_pct": 0.0,
                    "range_position_pct": 0.5,
                    "mid_range": True,
                },
            }

        range_size = upper_bound - lower_bound
        current_price = last_close
        range_width_pct = (range_size / current_price) * 100 if current_price else 0.0
        range_position_pct = (current_price - lower_bound) / range_size
        tolerance = range_size * self.boundary_tolerance_ratio

        upper_touches = int((recent["high"] >= (upper_bound - tolerance)).sum())
        lower_touches = int((recent["low"] <= (lower_bound + tolerance)).sum())
        drift_pct = abs(last_close - first_close) / first_close * 100 if first_close else 0.0
        is_directionally_balanced = drift_pct <= max(range_width_pct * 0.35, 0.6)

        return {
            "upper_touches": upper_touches,
            "lower_touches": lower_touches,
            "is_directionally_balanced": is_directionally_balanced,
            "price_levels": {
                "range_upper_bound": upper_bound,
                "range_lower_bound": lower_bound,
                "range_width_pct": range_width_pct,
                "range_position_pct": range_position_pct,
                "near_lower_boundary": range_position_pct <= 0.2,
                "near_upper_boundary": range_position_pct >= 0.8,
                "mid_range": 0.35 <= range_position_pct <= 0.65,
                "range_upper_touches": upper_touches,
                "range_lower_touches": lower_touches,
                "range_drift_pct": drift_pct,
            },
        }
