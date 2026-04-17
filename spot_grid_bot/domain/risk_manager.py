from __future__ import annotations

from collections import deque

from domain.exposure import calculate_outstanding_exposure
from domain.models import Candle, DeRiskMode, IndicatorSnapshot, InventorySnapshot, LiveOrder, RegimeType, RiskDecision, RiskRuntimeState
from domain.strategy_config import StrategyConfig


class RiskManager:
    """Evaluate protective trading constraints and staged de-risk decisions."""

    def __init__(self, config: StrategyConfig) -> None:
        """Store strategy configuration used for runtime risk evaluation."""
        self.config = config

    def evaluate(
        self,
        candles: list[Candle],
        indicators: IndicatorSnapshot,
        inventory: InventorySnapshot,
        live_orders: list[LiveOrder],
        runtime_state: RiskRuntimeState,
        state_regime: RegimeType,
        volatility_cooldown_remaining: int = 0,
    ) -> RiskDecision:
        """Return the current risk decision for one symbol trading cycle."""
        reasons: list[str] = []
        can_trade = True
        pause_entries = False
        force_risk_off = False
        cancel_entries = False
        allow_exit_only = False
        de_risk_mode = DeRiskMode.NONE
        exposure = calculate_outstanding_exposure(live_orders)
        projected_inventory_notional = inventory.inventory_notional + exposure.outstanding_buy_notional
        projected_quote_usage = inventory.reserved_quote + exposure.outstanding_buy_notional
        symbol_inventory_cap = min(
            inventory.total_equity * self.config.risk.max_symbol_inventory_pct_of_equity,
            self.config.risk.max_symbol_notional_cap,
            self.config.risk.max_inventory_notional,
        )
        symbol_new_entry_cap = inventory.available_quote * self.config.risk.max_symbol_new_entry_pct_of_free_quote

        if self._breakout_detected(candles, indicators):
            reasons.append("breakout_kill_switch")
            force_risk_off = True
            cancel_entries = True
            allow_exit_only = True
            can_trade = False
            de_risk_mode = DeRiskMode.PANIC
            runtime_state.kill_switch_count += 1

        if indicators.atr_spike or indicators.abnormal_candle:
            reasons.append("abnormal_volatility_pause")
            pause_entries = True
            cancel_entries = True
            allow_exit_only = True
            if de_risk_mode == DeRiskMode.NONE:
                de_risk_mode = DeRiskMode.SOFT

        if inventory.inventory_notional >= symbol_inventory_cap:
            reasons.append("symbol_inventory_pct_limit")
            pause_entries = True
            allow_exit_only = True
            de_risk_mode = _max_de_risk_mode(de_risk_mode, DeRiskMode.HARD)

        if symbol_new_entry_cap < self.config.risk.min_symbol_entry_notional:
            reasons.append("symbol_free_quote_entry_limit")
            pause_entries = True

        projected_inventory_cap = symbol_inventory_cap * self.config.risk.projected_inventory_buffer_pct
        if projected_inventory_notional >= projected_inventory_cap:
            reasons.append("projected_inventory_limit")
            pause_entries = True
            allow_exit_only = True
            de_risk_mode = _max_de_risk_mode(de_risk_mode, DeRiskMode.SOFT)

        projected_quote_cap = max(inventory.total_equity * self.config.risk.projected_quote_usage_pct, self.config.risk.min_quote_balance)
        if projected_quote_usage >= projected_quote_cap:
            reasons.append("projected_quote_usage_limit")
            pause_entries = True

        if inventory.available_quote < self.config.risk.min_quote_balance:
            reasons.append("quote_reserve_depleted")
            pause_entries = True

        if self._daily_drawdown_exceeded(runtime_state, inventory.total_equity):
            reasons.append("daily_drawdown_pause")
            force_risk_off = True
            allow_exit_only = True
            cancel_entries = True
            can_trade = False
            de_risk_mode = _max_de_risk_mode(de_risk_mode, DeRiskMode.HARD)

        if indicators.atr14 > 0 and indicators.atr_spike and indicators.realized_volatility > self.config.risk.emergency_volatility_multiplier * 0.01:
            reasons.append("emergency_volatility")
            force_risk_off = True
            allow_exit_only = True
            cancel_entries = True
            can_trade = False
            de_risk_mode = _max_de_risk_mode(de_risk_mode, DeRiskMode.PANIC)

        if volatility_cooldown_remaining > 0:
            reasons.append("volatility_cooldown_active")
            pause_entries = True
            cancel_entries = True

        if state_regime == RegimeType.DOWNTREND:
            reasons.append("downtrend_no_new_buys")
            pause_entries = True
            de_risk_mode = _max_de_risk_mode(de_risk_mode, DeRiskMode.SOFT)

        if state_regime == RegimeType.RISK_OFF:
            reasons.append("state_risk_off")
            force_risk_off = True
            allow_exit_only = True
            cancel_entries = True
            can_trade = False
            de_risk_mode = _max_de_risk_mode(de_risk_mode, DeRiskMode.PANIC)

        return RiskDecision(
            can_trade=can_trade and not force_risk_off,
            pause_entries=pause_entries or force_risk_off,
            force_risk_off=force_risk_off,
            cancel_entries=cancel_entries,
            allow_exit_only=allow_exit_only or force_risk_off,
            de_risk_mode=de_risk_mode,
            outstanding_buy_notional=exposure.outstanding_buy_notional,
            projected_inventory_notional=projected_inventory_notional,
            projected_quote_usage=projected_quote_usage,
            reasons=reasons,
        )

    def _breakout_detected(self, candles: list[Candle], indicators: IndicatorSnapshot) -> bool:
        """Check whether the latest close broke out beyond the configured ATR buffer."""
        if len(candles) <= self.config.risk.breakout_lookback:
            return False
        recent = candles[-self.config.risk.breakout_lookback - 1 : -1]
        breakout_high = max(candle.high for candle in recent) + indicators.atr14 * self.config.risk.breakout_atr_buffer
        breakout_low = min(candle.low for candle in recent) - indicators.atr14 * self.config.risk.breakout_atr_buffer
        close = candles[-1].close
        return close > breakout_high or close < breakout_low

    def _daily_drawdown_exceeded(self, runtime_state: RiskRuntimeState, equity: float) -> bool:
        """Track recent equity and return whether the configured drawdown pause was exceeded."""
        recent_equity = deque(runtime_state.recent_equity, maxlen=48)
        recent_equity.append(equity)
        runtime_state.recent_equity = list(recent_equity)
        peak = max(recent_equity, default=equity)
        if peak <= 0:
            return False
        drawdown = (peak - equity) / peak
        return drawdown >= self.config.risk.daily_drawdown_pause_pct


def _max_de_risk_mode(left: DeRiskMode, right: DeRiskMode) -> DeRiskMode:
    """Return the more aggressive of two de-risk modes."""
    return left if _de_risk_rank(left) >= _de_risk_rank(right) else right


def _de_risk_rank(mode: DeRiskMode) -> int:
    """Map staged de-risk modes to monotonic severity ranks."""
    return {
        DeRiskMode.NONE: 0,
        DeRiskMode.SOFT: 1,
        DeRiskMode.HARD: 2,
        DeRiskMode.PANIC: 3,
    }[mode]
