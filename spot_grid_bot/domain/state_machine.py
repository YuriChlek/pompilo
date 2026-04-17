from __future__ import annotations

from domain.models import RegimeSnapshot, RegimeType, StrategyState
from domain.strategy_config import StrategyConfig


class StrategyStateMachine:
    def __init__(self, config: StrategyConfig, initial_state: StrategyState) -> None:
        self.config = config
        self.state = initial_state

    def on_bar(self, regime_snapshot: RegimeSnapshot) -> StrategyState:
        state = self.state
        state.bars_in_state += 1
        if state.cooldown_remaining > 0:
            state.cooldown_remaining -= 1
        if state.volatility_cooldown_remaining > 0:
            state.volatility_cooldown_remaining -= 1

        if regime_snapshot.regime == RegimeType.HIGH_VOLATILITY:
            state.volatility_cooldown_remaining = max(
                state.volatility_cooldown_remaining,
                self.config.risk.volatility_cooldown_bars,
            )

        if regime_snapshot.regime == state.regime:
            state.pending_regime = None
            state.pending_count = 0
            return state

        if state.cooldown_remaining > 0:
            return state

        if state.pending_regime != regime_snapshot.regime:
            state.pending_regime = regime_snapshot.regime
            state.pending_count = 1
            return state

        state.pending_count += 1
        if state.pending_count >= self.config.regime.hysteresis_confirm_bars:
            state.regime = regime_snapshot.regime
            state.bars_in_state = 0
            state.cooldown_remaining = self.config.regime.state_cooldown_bars
            state.pending_regime = None
            state.pending_count = 0
        return state
