from __future__ import annotations

import logging

from domain.models import ExecutionResult, SpotSignal
from utils.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class LoggingSignalNotifier:
    """Log the result of processing one trading signal."""

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


class TelegramSignalNotifier:
    """Send selected execution notifications to Telegram."""

    def __init__(self, token: str = TELEGRAM_BOT_TOKEN, chat_id: str = TELEGRAM_CHAT_ID) -> None:
        self.token = token
        self.chat_id = chat_id

    async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
        if not self.token or not self.chat_id:
            return
        should_notify = (
            result.executed
            or (result.dry_run and result.action != "skip")
            or result.reason in {"no_loss_guard_infrastructure", "duplicate_signal_order"}
        )
        if not should_notify:
            return
        try:
            from telegram import Bot
        except ModuleNotFoundError:
            logger.warning("telegram_unavailable reason=python_telegram_bot_missing")
            return

        message = (
            f"spot_signal_processed\n"
            f"symbol={signal.symbol}\n"
            f"signal={signal.signal_type}\n"
            f"action={result.action}\n"
            f"executed={result.executed}\n"
            f"dry_run={result.dry_run}\n"
            f"reason={result.reason}\n"
            f"signal_price={signal.signal_price}\n"
            f"executed_price={result.executed_price}"
        )
        try:
            await Bot(token=self.token).send_message(chat_id=self.chat_id, text=message)
        except Exception:
            logger.exception("telegram_notification_failed symbol=%s action=%s", signal.symbol, result.action)


class CompositeSignalNotifier:
    """Fan out one signal notification to multiple notifiers."""

    def __init__(self, *notifiers) -> None:
        self.notifiers = notifiers

    async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
        for notifier in self.notifiers:
            await notifier.notify(signal, result)
