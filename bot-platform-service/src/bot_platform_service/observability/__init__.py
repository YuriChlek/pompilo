"""Observability helpers for Bot Platform Service."""

from bot_platform_service.observability.alerts import Alert, AlertRuleEvaluator, AlertThresholds
from bot_platform_service.observability.dashboards import DashboardDefinition, DashboardPanel, baseline_dashboard
from bot_platform_service.observability.health import (
    HealthEndpoint,
    OperationalIssue,
    OperationalStatusProvider,
    PlatformHealthReport,
    PlatformHealthService,
    instance_issue,
    module_issue,
    run_issue,
)
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
    InMemoryMetricsRecorder,
    MetricSample,
)
from bot_platform_service.observability.structured_logging import JsonStructuredLogger, StructuredLogRecord

__all__ = [
    "BOT_PLATFORM_ACTIVE_INSTANCES",
    "BOT_PLATFORM_EVENT_LAG_SECONDS",
    "BOT_PLATFORM_HEALTH_STATUS",
    "BOT_PLATFORM_MODULE_HEALTH",
    "BOT_PLATFORM_RUN_DURATION_SECONDS",
    "BOT_PLATFORM_RUN_FAILURES_TOTAL",
    "BOT_PLATFORM_RUNS_TOTAL",
    "BOT_PLATFORM_SNAPSHOT_LAG_SECONDS",
    "BOT_PLATFORM_SIGNAL_PUBLISH_FAILURES_TOTAL",
    "BOT_PLATFORM_SIGNALS_TOTAL",
    "Alert",
    "AlertRuleEvaluator",
    "AlertThresholds",
    "DashboardDefinition",
    "DashboardPanel",
    "HealthEndpoint",
    "InMemoryMetricsRecorder",
    "JsonStructuredLogger",
    "MetricSample",
    "OperationalIssue",
    "OperationalStatusProvider",
    "PlatformHealthReport",
    "PlatformHealthService",
    "StructuredLogRecord",
    "baseline_dashboard",
    "instance_issue",
    "module_issue",
    "run_issue",
]
