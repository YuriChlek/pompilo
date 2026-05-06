from __future__ import annotations

from dataclasses import dataclass

from domain.execution import decide_spot_execution
from domain.models import ExecutionDecision, PositionState, SpotSignal
from domain.signals import generate_spot_signal


@dataclass(frozen=True)
class SpotTradingPlan:
    """Combined output of Greenwich signal generation and execution planning."""

    signal: SpotSignal
    decision: ExecutionDecision


class GreenwichSpotPlanner:
    """Planner facade for the current Greenwich-based spot strategy."""

    def plan(
        self,
        symbol: str,
        candles_df,
        position_state: PositionState,
        available_quote_balance,
    ) -> SpotTradingPlan:
        """Build both the signal and the execution decision for one symbol."""

        signal = generate_spot_signal(symbol, candles_df)
        decision = decide_spot_execution(signal, position_state, available_quote_balance)
        return SpotTradingPlan(signal=signal, decision=decision)


__all__ = ["GreenwichSpotPlanner", "SpotTradingPlan"]
