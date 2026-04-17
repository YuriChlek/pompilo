from __future__ import annotations

import logging

from trading.domain.models import ExecutionResult, SpotSignal

logger = logging.getLogger(__name__)


class LoggingSignalNotifier:
    async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
        logger.info(
            "spot_signal_processed symbol=%s signal=%s action=%s executed=%s dry_run=%s reason=%s signal_price=%s executed_price=%s",
            signal.symbol,
            signal.signal_type,
            result.action,
            result.executed,
            result.dry_run,
            result.reason,
            signal.signal_price,
            result.executed_price,
        )
