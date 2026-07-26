from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from market_data_service.observability.metrics import (
    MARKET_DATA_GAP_COUNT,
    MARKET_DATA_OUTBOX_LAG_SECONDS,
    MARKET_DATA_PROVIDER_ERRORS_TOTAL,
    MARKET_DATA_SNAPSHOT_AGE_SECONDS,
    MARKET_DATA_SYNC_JOBS_STUCK_TOTAL,
    MetricSample,
)


@dataclass(frozen=True, slots=True)
class AlertRule:
    name: str
    metric_name: str
    threshold: float
    severity: str
    description: str


@dataclass(frozen=True, slots=True)
class AlertState:
    name: str
    metric_name: str
    value: float
    threshold: float
    severity: str
    labels: dict[str, str]
    description: str


DEFAULT_ALERT_RULES = (
    AlertRule(
        name="market_data_snapshot_stale",
        metric_name=MARKET_DATA_SNAPSHOT_AGE_SECONDS,
        threshold=3900.0,
        severity="warning",
        description="snapshot age is above the configured 1h freshness threshold",
    ),
    AlertRule(
        name="market_data_gap_detected",
        metric_name=MARKET_DATA_GAP_COUNT,
        threshold=0.0,
        severity="warning",
        description="one or more expected candles are missing from a validated range",
    ),
    AlertRule(
        name="market_data_provider_errors_high",
        metric_name=MARKET_DATA_PROVIDER_ERRORS_TOTAL,
        threshold=5.0,
        severity="critical",
        description="provider errors exceeded the baseline threshold",
    ),
    AlertRule(
        name="market_data_outbox_lag_high",
        metric_name=MARKET_DATA_OUTBOX_LAG_SECONDS,
        threshold=300.0,
        severity="warning",
        description="outbox lag is above the publish threshold",
    ),
    AlertRule(
        name="market_data_sync_jobs_stuck",
        metric_name=MARKET_DATA_SYNC_JOBS_STUCK_TOTAL,
        threshold=0.0,
        severity="critical",
        description="sync jobs are stuck in RUNNING status",
    ),
)


def evaluate_alerts(
    samples: Iterable[MetricSample],
    *,
    rules: Iterable[AlertRule] = DEFAULT_ALERT_RULES,
) -> list[AlertState]:
    rules_by_metric = {rule.metric_name: rule for rule in rules}
    alerts: list[AlertState] = []
    for sample in samples:
        rule = rules_by_metric.get(sample.name)
        if rule is None or sample.value <= rule.threshold:
            continue
        alerts.append(
            AlertState(
                name=rule.name,
                metric_name=sample.name,
                value=sample.value,
                threshold=rule.threshold,
                severity=rule.severity,
                labels=dict(sample.labels),
                description=rule.description,
            )
        )
    return alerts
