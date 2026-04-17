from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


def _default_cluster_map() -> dict[str, str]:
    """Return the default correlation cluster mapping for the supported futures symbols."""
    return {
        "ETHUSDT": "majors",
        "XRPUSDT": "majors",
        "LTCUSDT": "majors",
        "SOLUSDT": "l1_l2_beta",
        "AVAXUSDT": "l1_l2_beta",
        "APTUSDT": "l1_l2_beta",
        "SUIUSDT": "l1_l2_beta",
        "ADAUSDT": "l1_l2_beta",
        "ARBUSDT": "l1_l2_beta",
        "NEARUSDT": "l1_l2_beta",
        "DOTUSDT": "l1_l2_beta",
        "TONUSDT": "l1_l2_beta",
        "TAOUSDT": "l1_l2_beta",
        "DOGEUSDT": "meme_beta",
        "WIFUSDT": "meme_beta",
        "PENGUUSDT": "meme_beta",
        "VIRTUALUSDT": "meme_beta",
        "WLDUSDT": "meme_beta",
        "AAVEUSDT": "infra",
        "LINKUSDT": "infra",
        "UNIUSDT": "infra",
        "ENAUSDT": "infra",
        "JUPUSDT": "infra",
        "ZECUSDT": "alts",
    }


@dataclass(frozen=True)
class RiskStrategyConfig:
    """Configuration for position sizing and entry tolerances."""

    primary_risk_pct: Decimal = Decimal("0.5")
    supertrend_tolerance: Decimal = Decimal("0.005")


@dataclass(frozen=True)
class ExitStrategyConfig:
    """Configuration for live exit management after a position is opened."""

    tp1_r_multiple: Decimal = Decimal("1.0")
    tp1_close_fraction: Decimal = Decimal("50")
    breakeven_trigger_r: Decimal = Decimal("1.5")
    trail_activation_r: Decimal = Decimal("2.0")
    trail_atr_multiple: Decimal = Decimal("1.0")
    allow_regime_exit: bool = False


@dataclass(frozen=True)
class RegimeStrategyConfig:
    """Configuration for filtering breakout entries by market regime."""

    require_h4_alignment: bool = True
    require_d1_alignment: bool = False
    allow_high_vol_entries: bool = False
    range_confidence_threshold: Decimal = Decimal("65")
    max_atr_pct: Decimal = Decimal("3.5")
    min_trend_strength: str = "medium"


@dataclass(frozen=True)
class BreakoutTrendStrategyConfig:
    """Configuration for trend-following breakout entries on candle close."""

    enabled: bool = True
    lookback_candles: int = 20
    min_volume_spike_ratio: Decimal = Decimal("1.20")
    strong_trend_volume_ratio: Decimal = Decimal("1.05")
    medium_trend_volume_ratio: Decimal = Decimal("1.15")
    weak_trend_allows_entry: bool = False
    stop_atr_multiplier: Decimal = Decimal("1.30")
    reclaim_stop_atr_multiplier: Decimal = Decimal("1.00")
    take_profit_r: Decimal = Decimal("3.00")
    max_breakout_candle_atr: Decimal = Decimal("2.50")
    reclaim_max_candle_atr: Decimal = Decimal("2.00")
    min_breakout_close_pct: Decimal = Decimal("0.0005")
    atr_breakout_buffer_fraction: Decimal = Decimal("0.10")
    reclaim_enabled: bool = True
    reclaim_tolerance_atr_fraction: Decimal = Decimal("0.15")
    require_reclaim_close_in_breakout_direction: bool = True
    allow_h1_range_in_strong_h4_trend: bool = True
    high_vol_risk_multiplier: Decimal = Decimal("0.50")
    order_type: str = "Market"


@dataclass(frozen=True)
class FundingFilterConfig:
    """Configuration for optionally skipping crowded perpetual-futures breakouts."""

    enabled: bool = False
    long_funding_cap: Decimal = Decimal("0.0004")
    short_funding_floor: Decimal = Decimal("-0.0004")


@dataclass(frozen=True)
class PortfolioRiskConfig:
    """Configuration for portfolio-wide entry admission controls."""

    enabled: bool = True
    assumed_equity: Decimal = Decimal("1000")
    max_open_positions: int = 3
    max_portfolio_heat_pct: Decimal = Decimal("2.0")
    max_positions_per_cluster: int = 2
    daily_loss_stop_r: Decimal = Decimal("2.0")
    cluster_map: dict[str, str] = field(default_factory=_default_cluster_map)


@dataclass(frozen=True)
class StrategyConfig:
    """Top-level trading strategy configuration shared across layers."""

    risk: RiskStrategyConfig = field(default_factory=RiskStrategyConfig)
    exit: ExitStrategyConfig = field(default_factory=ExitStrategyConfig)
    regime: RegimeStrategyConfig = field(default_factory=RegimeStrategyConfig)
    breakout: BreakoutTrendStrategyConfig = field(default_factory=BreakoutTrendStrategyConfig)
    funding: FundingFilterConfig = field(default_factory=FundingFilterConfig)
    portfolio: PortfolioRiskConfig = field(default_factory=PortfolioRiskConfig)


DEFAULT_STRATEGY_CONFIG = StrategyConfig()
