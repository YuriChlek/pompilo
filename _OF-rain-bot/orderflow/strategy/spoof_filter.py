from __future__ import annotations

from orderflow.market_data.models import LiquidityWall
from utils.config import (
    ORDERFLOW_MAX_CHASE_TICKS,
    ORDERFLOW_MAX_PULL_EVENTS,
    ORDERFLOW_MAX_SPOOF_SCORE,
    ORDERFLOW_MIN_DEFENDED_RATIO,
)


class SpoofFilter:
    def is_valid(self, wall: LiquidityWall) -> bool:
        if wall.pull_count > ORDERFLOW_MAX_PULL_EVENTS:
            return False
        if wall.chase_count > ORDERFLOW_MAX_CHASE_TICKS:
            return False
        if wall.spoof_risk_score > ORDERFLOW_MAX_SPOOF_SCORE:
            return False
        if float(wall.metadata.get("retained_ratio", 0.0)) < ORDERFLOW_MIN_DEFENDED_RATIO:
            return False
        return True
