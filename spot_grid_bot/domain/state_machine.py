from __future__ import annotations

from dataclasses import replace

from domain.models import RegimeSnapshot, RegimeType, StrategyState
from domain.strategy_config import StrategyConfig


class StrategyStateMachine:
    def __init__(self, config: StrategyConfig, initial_state: StrategyState) -> None:
        self.config = config
        self.state = initial_state

    def on_bar(self, regime_snapshot: RegimeSnapshot) -> StrategyState:
        state = self.state
        updated = replace(
            state,
            bars_in_state=state.bars_in_state + 1,
            cooldown_remaining=max(state.cooldown_remaining - 1, 0),
            volatility_cooldown_remaining=max(state.volatility_cooldown_remaining - 1, 0),
        )

        if regime_snapshot.regime == RegimeType.HIGH_VOLATILITY:
            updated = replace(
                updated,
                volatility_cooldown_remaining=max(
                    updated.volatility_cooldown_remaining,
                    self.config.risk.volatility_cooldown_bars,
                ),
            )

        if regime_snapshot.regime == updated.regime:
            updated = replace(updated, pending_regime=None, pending_count=0)
            self.state = updated
            return updated

        if updated.cooldown_remaining > 0:
            self.state = updated
            return updated

        if updated.pending_regime != regime_snapshot.regime:
            updated = replace(updated, pending_regime=regime_snapshot.regime, pending_count=1)
            self.state = updated
            return updated

        updated = replace(updated, pending_count=updated.pending_count + 1)
        if updated.pending_count >= self.config.regime.hysteresis_confirm_bars:
            updated = replace(
                updated,
                regime=regime_snapshot.regime,
                bars_in_state=0,
                cooldown_remaining=self.config.regime.state_cooldown_bars,
                pending_regime=None,
                pending_count=0,
            )

        self.state = updated
        return updated
