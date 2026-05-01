from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from urllib import error, request

from domain.models import DeRiskMode, StrategyDecision

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


@dataclass(slots=True)
class TelegramNotifierConfig:
    """Configuration needed for future Telegram notifier integration."""

    bot_token: str
    chat_id: str
    timeout_seconds: float = 10.0


class NotificationSeverity(str, Enum):
    """Severity bucket for outbound operator notifications."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class TelegramSignalNotifier:
    """Best-effort Telegram notifier for critical trading rebuild events."""

    def __init__(self, config: TelegramNotifierConfig) -> None:
        self.config = config
        self._fallback = LoggingSignalNotifier()

    async def notify_rebuild(self, decision: StrategyDecision) -> None:
        """Send critical rebuild events to Telegram without affecting trading flow on failure."""
        if self._should_alert(decision):
            message = self._format_message(decision)
            try:
                await asyncio.to_thread(self._send_message, message)
            except Exception as exc:
                logger.exception(
                    "telegram_notifier_send_failed symbol=%s error_type=%s",
                    decision.symbol,
                    type(exc).__name__,
                )
        await self._fallback.notify_rebuild(decision)

    def _should_alert(self, decision: StrategyDecision) -> bool:
        """Return whether this rebuild warrants an operator-facing Telegram alert."""
        risk = decision.risk
        critical_reasons = {
            "breakout_kill_switch",
            "daily_drawdown_pause",
            "emergency_volatility",
            "state_risk_off",
        }
        return (
            risk.force_risk_off
            or risk.de_risk_mode in {DeRiskMode.HARD, DeRiskMode.PANIC}
            or decision.kill_switch_count > 0
            or any(reason in critical_reasons for reason in risk.reasons)
        )

    def _severity(self, decision: StrategyDecision) -> NotificationSeverity:
        """Map a strategy decision to a coarse operator notification severity."""
        if decision.risk.de_risk_mode == DeRiskMode.PANIC or "emergency_volatility" in decision.risk.reasons:
            return NotificationSeverity.CRITICAL
        if (
            decision.risk.force_risk_off
            or decision.risk.de_risk_mode == DeRiskMode.HARD
            or "daily_drawdown_pause" in decision.risk.reasons
            or "breakout_kill_switch" in decision.risk.reasons
            or "state_risk_off" in decision.risk.reasons
        ):
            return NotificationSeverity.CRITICAL
        return NotificationSeverity.WARNING

    def _format_message(self, decision: StrategyDecision) -> str:
        """Build a compact plain-text Telegram message for a critical rebuild event."""
        severity = self._severity(decision).value
        risk_reasons = ",".join(decision.risk.reasons) if decision.risk.reasons else "none"
        reasons = ",".join(decision.reasons) if decision.reasons else "none"
        return (
            f"[{severity}] Spot Grid Bot\n"
            f"Symbol: {decision.symbol}\n"
            f"Regime: {decision.regime.value}\n"
            f"De-risk: {decision.risk.de_risk_mode.value}\n"
            f"Force risk off: {decision.risk.force_risk_off}\n"
            f"Target orders: {len(decision.target_orders)}\n"
            f"Target diff: {decision.target_order_diff_count}\n"
            f"Kill switch count: {decision.kill_switch_count}\n"
            f"Risk reasons: {risk_reasons}\n"
            f"Reasons: {reasons}"
        )

    def _send_message(self, message: str) -> None:
        """Send one Telegram Bot API message using the standard library HTTP client."""
        payload = json.dumps(
            {
                "chat_id": self.config.chat_id,
                "text": message,
                "disable_web_page_preview": True,
            }
        ).encode("utf-8")
        req = request.Request(
            url=f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status >= 400:
                    raise RuntimeError(f"Telegram API returned HTTP {status}")
        except error.HTTPError as exc:
            raise RuntimeError(f"Telegram API returned HTTP {exc.code}") from exc
