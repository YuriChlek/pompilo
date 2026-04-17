from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RangeMarketResult:
    """Describe the current range-market assessment for one timeframe."""

    is_range: bool
    confidence: float
    reasons: List[str]
    atr_pct: float
    fractal_pattern: str
    avg_upper: float
    avg_lower: float
    range_type: str  # 'upper_cluster', 'lower_cluster', 'both_cluster', 'range'
    cluster_strength: str  # 'weak', 'medium', 'strong'
    price_levels: Dict[str, float]
    upper_fractals_analyzed: int
    lower_fractals_analyzed: int

    @classmethod
    def empty(cls) -> "RangeMarketResult":
        """Return an empty range-analysis result used as a safe default."""
        return cls(
            is_range=False,
            confidence=0.0,
            reasons=[],
            atr_pct=0.0,
            fractal_pattern="unknown",
            avg_upper=0.0,
            avg_lower=0.0,
            range_type="none",
            cluster_strength="weak",
            price_levels={},
            upper_fractals_analyzed=0,
            lower_fractals_analyzed=0,
        )


@dataclass
class FractalResult:
    """Store full and recent upper/lower fractal collections."""

    upper_fractals: List[Dict[str, Any]]
    lower_fractals: List[Dict[str, Any]]
    last_3_upper_fractals: List[Dict[str, Any]]
    last_3_lower_fractals: List[Dict[str, Any]]


@dataclass
class TrendResult:
    """Bundle the latest multi-timeframe indicator state for signal generation."""

    atr: float
    super_trend: float
    super_trend_signal: str
    super_trend_h4: float
    super_trend_h4_signal: str
    super_trend_d1: float
    super_trend_d1_signal: str
    mfi: float
    mfi_signal: str
    candle: Dict[str, Any]
    indicators: Dict[str, Any]
    gmma_analysis: Optional[Dict[str, Any]] = None
    volume_analysis: Optional[Dict[str, Any]] = None
    cvd_analysis: Optional[Dict[str, Any]] = None
    timestamp: Optional[pd.Timestamp] = None
    fractal_analysis_h1: Optional[FractalResult] = None
    fractal_analysis_h4: Optional[FractalResult] = None
    fractal_analysis_d1: Optional[FractalResult] = None
    range_analysis_h1: Optional[RangeMarketResult] = None
    range_analysis_h4: Optional[RangeMarketResult] = None
    range_analysis_d1: Optional[RangeMarketResult] = None
    funding_rate: Optional[float] = None
