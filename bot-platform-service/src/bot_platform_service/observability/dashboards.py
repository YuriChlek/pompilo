from __future__ import annotations

from dataclasses import dataclass

from bot_platform_service.observability.metrics import (
    BOT_PLATFORM_ACTIVE_INSTANCES,
    BOT_PLATFORM_EVENT_LAG_SECONDS,
    BOT_PLATFORM_HEALTH_STATUS,
    BOT_PLATFORM_MODULE_HEALTH,
    BOT_PLATFORM_RUN_DURATION_SECONDS,
    BOT_PLATFORM_RUN_FAILURES_TOTAL,
    BOT_PLATFORM_RUNS_TOTAL,
    BOT_PLATFORM_SNAPSHOT_LAG_SECONDS,
    BOT_PLATFORM_SIGNAL_PUBLISH_FAILURES_TOTAL,
    BOT_PLATFORM_SIGNALS_TOTAL,
)


@dataclass(frozen=True, slots=True)
class DashboardPanel:
    """Dashboard panel metadata for baseline operations."""

    title: str
    metric_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DashboardDefinition:
    """Dashboard definition independent from a concrete dashboard vendor."""

    title: str
    panels: tuple[DashboardPanel, ...]


def baseline_dashboard() -> DashboardDefinition:
    """Return baseline dashboard panels for Bot Platform operators."""

    return DashboardDefinition(
        title="Bot Platform Baseline",
        panels=(
            DashboardPanel("Runs", (BOT_PLATFORM_RUNS_TOTAL, BOT_PLATFORM_RUN_FAILURES_TOTAL, BOT_PLATFORM_RUN_DURATION_SECONDS)),
            DashboardPanel("Instances", (BOT_PLATFORM_ACTIVE_INSTANCES,)),
            DashboardPanel("Signals", (BOT_PLATFORM_SIGNALS_TOTAL, BOT_PLATFORM_SIGNAL_PUBLISH_FAILURES_TOTAL)),
            DashboardPanel("Lag", (BOT_PLATFORM_SNAPSHOT_LAG_SECONDS, BOT_PLATFORM_EVENT_LAG_SECONDS)),
            DashboardPanel("Health", (BOT_PLATFORM_HEALTH_STATUS, BOT_PLATFORM_MODULE_HEALTH)),
        ),
    )


__all__ = ["DashboardDefinition", "DashboardPanel", "baseline_dashboard"]
