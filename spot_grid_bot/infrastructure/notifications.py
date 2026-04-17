from __future__ import annotations

import logging

from domain.models import StrategyDecision

logger = logging.getLogger(__name__)


class LoggingSignalNotifier:
    """Notification adapter that logs rebuild events instead of sending external messages."""

    async def notify_rebuild(self, decision: StrategyDecision) -> None:
        """Log a summary line for a completed grid rebuild decision."""
        logger.info(
            "grid_rebuilt symbol=%s regime=%s target_orders=%s reasons=%s",
            decision.symbol,
            decision.regime.value,
            len(decision.target_orders),
            ",".join(decision.reasons),
        )
