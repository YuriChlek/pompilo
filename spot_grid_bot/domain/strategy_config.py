from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketDataConfig:
    candle_interval: str = "1h"
    candles_lookback: int = 2400
    candle_schema: str = "_candles_trading_data"
    candle_table_suffix: str = "_p_candles"


@dataclass(frozen=True)
class RegimeConfig:
    ema_fast_length: int = 20
    ema_mid_length: int = 50
    ema_slow_length: int = 200
    atr_length: int = 14
    realized_vol_length: int = 20
    regime_lookback: int = 30
    range_width_atr_min: float = 1.2
    range_width_atr_max: float = 5.0
    range_directional_threshold: float = 1.4
    ema_mid_slope_flat_threshold: float = 0.0010
    ema_mid_slope_trend_threshold: float = 0.0015
    atr_spike_multiplier: float = 1.8
    abnormal_candle_atr_multiplier: float = 2.2
    state_cooldown_bars: int = 4
    hysteresis_confirm_bars: int = 2
    structure_swing_window: int = 2
    structure_lookback: int = 40


@dataclass(frozen=True)
class GridConfig:
    atr_grid_step_multiplier: float = 0.6
    range_grid_width_multiplier: float = 2.2
    trend_grid_width_multiplier: float = 1.6
    range_buy_fraction: float = 0.45
    min_grid_levels: int = 3
    max_grid_levels: int = 7
    max_active_levels: int = 5
    pullback_reentry_atr: float = 0.7
    rebalance_profit_target_atr: float = 0.8
    uptrend_buy_levels: int = 3
    uptrend_pullback_atr_multipliers: tuple[float, ...] = (0.5, 1.0, 1.5)
    uptrend_buy_size_weights: tuple[float, ...] = (0.2, 0.3, 0.5)
    uptrend_sell_levels: int = 3
    uptrend_sell_atr_multipliers: tuple[float, ...] = (1.0, 1.75, 2.5)
    uptrend_sell_size_weights: tuple[float, ...] = (0.2, 0.3, 0.5)
    uptrend_allocation_weight: float = 1.25
    uptrend_underwater_budget_penalty: float = 0.6
    uptrend_underwater_max_buy_levels: int = 2
    uptrend_max_price_extension_from_ema20_bps: float = 150.0
    underwater_averaging_enabled: bool = True
    underwater_averaging_trigger_pct: float = 0.10
    underwater_recovery_budget_pct: float = 0.30
    underwater_range_budget_multiplier: float = 0.50
    underwater_uptrend_budget_multiplier: float = 1.00
    underwater_max_recovery_levels: int = 2
    underwater_deep_stop_pct: float = 0.25
    range_entry_quality_soft_threshold: float = 0.35
    range_entry_quality_hard_threshold: float = 0.20
    range_weak_entry_budget_penalty: float = 0.90
    range_poor_entry_budget_penalty: float = 0.75
    range_weak_max_buy_levels: int = 4
    range_poor_max_buy_levels: int = 3
    range_entry_upper_band_soft_limit: float = 0.82
    range_breakdown_directional_threshold_multiplier: float = 1.0
    underwater_recovery_sell_aggressiveness_atr: float = 0.35
    uptrend_strong_trend_threshold: float = 0.70
    uptrend_weak_trend_threshold: float = 0.45
    uptrend_adaptive_take_profit_bonus_atr: float = 0.35
    uptrend_adaptive_take_profit_penalty_atr: float = 0.20


@dataclass(frozen=True)
class RiskConfig:
    breakout_lookback: int = 20
    breakout_atr_buffer: float = 0.6
    max_inventory_notional: float = 12_000.0
    max_notional_per_level: float = 1_250.0
    global_quote_reserve_pct: float = 0.25
    global_max_new_entry_pct_of_free_quote: float = 0.30
    max_symbol_inventory_pct_of_equity: float = 0.08
    max_symbol_new_entry_pct_of_free_quote: float = 0.10
    max_symbol_notional_cap: float = 400.0
    min_symbol_entry_notional: float = 3.0
    global_max_portfolio_inventory_pct_of_equity: float = 0.60
    max_concurrent_entry_symbols: int = 4
    allocation_weight_range: float = 0.8
    allocation_weight_uptrend: float = 1.0
    min_quote_balance: float = 100.0
    daily_drawdown_pause_pct: float = 0.06
    emergency_volatility_multiplier: float = 2.5
    force_exit_slippage_bps: float = 12.0
    volatility_cooldown_bars: int = 3
    projected_inventory_buffer_pct: float = 1.10
    projected_quote_usage_pct: float = 0.85


@dataclass(frozen=True)
class ExecutionConfig:
    price_tick: float = 0.1
    size_step: float = 0.0001
    min_order_size: float = 0.0001
    min_order_notional_usdt: float = 2.0
    maker_fee_bps: float = 10.0
    rebuild_price_deviation_pct: float = 0.003
    target_price_diff_bps: float = 5.0
    target_size_diff_ratio: float = 0.05
    max_new_orders_per_cycle: int = 3
    max_cancel_replace_per_cycle: int = 6
    max_total_open_orders: int = 8
    min_level_distance_bps: float = 10.0
    min_buy_distance_from_live_bps: float = 30.0
    min_sell_markup_bps: float = 100.0


@dataclass(frozen=True)
class PortfolioConfig:
    base_asset: str = "BTC"
    quote_asset: str = "USDT"
    starting_quote_balance: float = 10_000.0
    starting_base_balance: float = 0.0


@dataclass(frozen=True)
class StrategyConfig:
    market_data: MarketDataConfig = MarketDataConfig()
    regime: RegimeConfig = RegimeConfig()
    grid: GridConfig = GridConfig()
    risk: RiskConfig = RiskConfig()
    execution: ExecutionConfig = ExecutionConfig()
    portfolio: PortfolioConfig = PortfolioConfig()


DEFAULT_STRATEGY_CONFIG = StrategyConfig()
