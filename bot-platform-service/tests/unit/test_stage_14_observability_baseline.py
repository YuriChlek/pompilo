from __future__ import annotations

import asyncio
import json
import logging

from bot_platform_service.domain import BotHealthStatus, BotInstanceStatus, BotModuleStatus, BotRunStatus
from bot_platform_service.observability import (
    BOT_PLATFORM_ACTIVE_INSTANCES,
    BOT_PLATFORM_EVENT_LAG_SECONDS,
    BOT_PLATFORM_RUNS_TOTAL,
    BOT_PLATFORM_SNAPSHOT_LAG_SECONDS,
    AlertThresholds,
    AlertRuleEvaluator,
    HealthEndpoint,
    InMemoryMetricsRecorder,
    JsonStructuredLogger,
    PlatformHealthService,
    StructuredLogRecord,
    baseline_dashboard,
    instance_issue,
    module_issue,
    run_issue,
)


class _HealthyProvider:
    async def failed_modules(self):
        return ()

    async def failed_instances(self):
        return ()

    async def failed_runs(self):
        return ()


class _FailedProvider:
    async def failed_modules(self):
        return (module_issue("spot_grid_bot", BotModuleStatus.DISABLED, "manual disable"),)

    async def failed_instances(self):
        return (instance_issue("instance-1", BotInstanceStatus.FAILED, "config invalid"),)

    async def failed_runs(self):
        return (run_issue("run-1", BotRunStatus.FAILED, "adapter error"),)


def test_health_endpoint_returns_healthy_status_when_no_failures() -> None:
    endpoint = HealthEndpoint(PlatformHealthService(_HealthyProvider()))

    status_code, payload = asyncio.run(endpoint.get())

    assert status_code == 200
    assert payload["status"] == BotHealthStatus.HEALTHY.value
    assert payload["failed_modules"] == []
    assert payload["failed_instances"] == []
    assert payload["failed_runs"] == []


def test_health_endpoint_exposes_failed_entities_and_reasons() -> None:
    endpoint = HealthEndpoint(PlatformHealthService(_FailedProvider()))

    status_code, payload = asyncio.run(endpoint.get())

    assert status_code == 503
    assert payload["status"] == BotHealthStatus.UNHEALTHY.value
    assert payload["failed_modules"][0]["entity_id"] == "spot_grid_bot"
    assert payload["failed_instances"][0]["reason"] == "config invalid"
    assert payload["failed_runs"][0]["reason"] == "adapter error"


def test_alert_rules_emit_operator_alerts_for_failed_entities() -> None:
    report = asyncio.run(PlatformHealthService(_FailedProvider()).report())

    alerts = AlertRuleEvaluator().evaluate(report)

    assert [alert.name for alert in alerts] == [
        "bot_platform_failed_modules",
        "bot_platform_failed_instances",
        "bot_platform_failed_runs",
    ]


def test_alert_rules_emit_threshold_alerts_for_operational_risks() -> None:
    alerts = AlertRuleEvaluator().evaluate_operational_metrics(
        run_failures=2,
        stuck_running_instances=1,
        snapshot_lag_seconds=301,
        event_lag_seconds=121,
        signal_publish_failures=1,
        thresholds=AlertThresholds(),
    )

    assert [alert.name for alert in alerts] == [
        "bot_platform_run_failures_total",
        "bot_platform_stuck_running_instances",
        "bot_platform_snapshot_lag_seconds",
        "bot_platform_event_lag_seconds",
        "bot_platform_signal_publish_failures_total",
    ]


def test_metrics_recorder_keeps_tagged_counters_and_observations() -> None:
    metrics = InMemoryMetricsRecorder()

    metrics.increment(BOT_PLATFORM_RUNS_TOTAL, tags={"module_id": "spot_grid_bot"})
    metrics.increment(BOT_PLATFORM_RUNS_TOTAL, tags={"module_id": "spot_grid_bot"})
    metrics.observe("bot_platform_run_duration_seconds", 1.25, tags={"status": "COMPLETE"})
    metrics.observe(BOT_PLATFORM_SNAPSHOT_LAG_SECONDS, 12, tags={"symbol": "ETHUSDT"})
    metrics.observe(BOT_PLATFORM_EVENT_LAG_SECONDS, 3, tags={"source": "market-data-service"})
    metrics.observe(BOT_PLATFORM_ACTIVE_INSTANCES, 2)

    assert metrics.counter_value(BOT_PLATFORM_RUNS_TOTAL, tags={"module_id": "spot_grid_bot"}) == 2
    snapshot = metrics.snapshot()
    assert any(sample.name == "bot_platform_run_duration_seconds" and sample.value == 1.25 for sample in snapshot)
    assert any(sample.name == BOT_PLATFORM_SNAPSHOT_LAG_SECONDS and sample.value == 12 for sample in snapshot)
    assert any(sample.name == BOT_PLATFORM_EVENT_LAG_SECONDS and sample.value == 3 for sample in snapshot)


def test_structured_log_record_is_json_serializable() -> None:
    payload = json.loads(StructuredLogRecord("INFO", "bot_run_completed", {"run_id": "run-1"}).to_json())

    assert payload["level"] == "INFO"
    assert payload["event"] == "bot_run_completed"
    assert payload["run_id"] == "run-1"


def test_json_structured_logger_writes_structured_message(caplog) -> None:
    logger = logging.getLogger("stage14")
    structured = JsonStructuredLogger(logger)

    with caplog.at_level(logging.INFO, logger="stage14"):
        structured.info("bot_instance_failed", instance_id="instance-1")

    assert "bot_instance_failed" in caplog.messages[0]
    assert "instance-1" in caplog.messages[0]


def test_baseline_dashboard_contains_run_signal_and_health_panels() -> None:
    dashboard = baseline_dashboard()

    assert dashboard.title == "Bot Platform Baseline"
    assert [panel.title for panel in dashboard.panels] == ["Runs", "Instances", "Signals", "Lag", "Health"]
