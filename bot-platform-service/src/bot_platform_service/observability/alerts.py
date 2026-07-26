from __future__ import annotations

from dataclasses import dataclass

from bot_platform_service.domain import BotHealthStatus
from bot_platform_service.observability.health import PlatformHealthReport
from bot_platform_service.observability.metrics import (
    BOT_PLATFORM_EVENT_LAG_SECONDS,
    BOT_PLATFORM_RUN_FAILURES_TOTAL,
    BOT_PLATFORM_SIGNAL_PUBLISH_FAILURES_TOTAL,
    BOT_PLATFORM_SNAPSHOT_LAG_SECONDS,
)


@dataclass(frozen=True, slots=True)
class Alert:
    """Operator alert emitted from health reports."""

    name: str
    severity: str
    message: str


@dataclass(frozen=True, slots=True)
class AlertThresholds:
    """Thresholds for baseline operational alert rules."""

    run_failures: float = 0
    stuck_running_instances: int = 0
    snapshot_lag_seconds: float = 300
    event_lag_seconds: float = 120
    signal_publish_failures: float = 0


class AlertRuleEvaluator:
    """Evaluate baseline alert rules for failed platform entities."""

    def evaluate(self, report: PlatformHealthReport) -> tuple[Alert, ...]:
        """Return alerts for failed modules, instances, or runs."""

        if report.status is BotHealthStatus.HEALTHY:
            return ()
        alerts: list[Alert] = []
        if report.failed_modules:
            alerts.append(Alert("bot_platform_failed_modules", "critical", f"{len(report.failed_modules)} module(s) require attention"))
        if report.failed_instances:
            alerts.append(Alert("bot_platform_failed_instances", "warning", f"{len(report.failed_instances)} instance(s) require attention"))
        if report.failed_runs:
            alerts.append(Alert("bot_platform_failed_runs", "warning", f"{len(report.failed_runs)} run(s) failed"))
        return tuple(alerts)

    def evaluate_operational_metrics(
        self,
        *,
        run_failures: float,
        stuck_running_instances: int,
        snapshot_lag_seconds: float,
        event_lag_seconds: float,
        signal_publish_failures: float,
        thresholds: AlertThresholds | None = None,
    ) -> tuple[Alert, ...]:
        """Return alerts for metric-threshold based production risks."""

        active_thresholds = thresholds or AlertThresholds()
        alerts: list[Alert] = []
        if run_failures > active_thresholds.run_failures:
            alerts.append(
                Alert(
                    BOT_PLATFORM_RUN_FAILURES_TOTAL,
                    "warning",
                    f"bot run failures above threshold: {run_failures:g}",
                )
            )
        if stuck_running_instances > active_thresholds.stuck_running_instances:
            alerts.append(
                Alert(
                    "bot_platform_stuck_running_instances",
                    "critical",
                    f"{stuck_running_instances} instance(s) stuck in RUNNING",
                )
            )
        if snapshot_lag_seconds > active_thresholds.snapshot_lag_seconds:
            alerts.append(
                Alert(
                    BOT_PLATFORM_SNAPSHOT_LAG_SECONDS,
                    "warning",
                    f"snapshot lag above threshold: {snapshot_lag_seconds:g}s",
                )
            )
        if event_lag_seconds > active_thresholds.event_lag_seconds:
            alerts.append(
                Alert(
                    BOT_PLATFORM_EVENT_LAG_SECONDS,
                    "warning",
                    f"event lag above threshold: {event_lag_seconds:g}s",
                )
            )
        if signal_publish_failures > active_thresholds.signal_publish_failures:
            alerts.append(
                Alert(
                    BOT_PLATFORM_SIGNAL_PUBLISH_FAILURES_TOTAL,
                    "critical",
                    f"signal publish failures above threshold: {signal_publish_failures:g}",
                )
            )
        return tuple(alerts)


__all__ = ["Alert", "AlertRuleEvaluator", "AlertThresholds"]
