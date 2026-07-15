from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType


BOT_PLATFORM_RUNS_TOTAL = "bot_platform_runs_total"
BOT_PLATFORM_RUN_DURATION_SECONDS = "bot_platform_run_duration_seconds"
BOT_PLATFORM_RUN_FAILURES_TOTAL = "bot_platform_run_failures_total"
BOT_PLATFORM_ACTIVE_INSTANCES = "bot_platform_active_instances"
BOT_PLATFORM_MODULE_HEALTH = "bot_platform_module_health"
BOT_PLATFORM_SIGNALS_TOTAL = "bot_platform_signals_total"
BOT_PLATFORM_SIGNAL_PUBLISH_FAILURES_TOTAL = "bot_platform_signal_publish_failures_total"
BOT_PLATFORM_SNAPSHOT_LAG_SECONDS = "bot_platform_snapshot_lag_seconds"
BOT_PLATFORM_EVENT_LAG_SECONDS = "bot_platform_event_lag_seconds"
BOT_PLATFORM_HEALTH_STATUS = "bot_platform_health_status"


@dataclass(frozen=True, slots=True)
class MetricSample:
    """Recorded metric sample with stable tags."""

    name: str
    value: float
    tags: MappingProxyType[str, str]


class InMemoryMetricsRecorder:
    """Small MetricsRecorder implementation suitable for tests and local health views."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._observations: list[MetricSample] = []

    def increment(self, name: str, *, tags: dict[str, str] | None = None) -> None:
        """Increment one counter metric."""

        key = (name, _tags_key(tags))
        self._counters[key] += 1

    def observe(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        """Record one observation sample."""

        self._observations.append(MetricSample(name=name, value=float(value), tags=MappingProxyType(dict(tags or {}))))

    def counter_value(self, name: str, *, tags: dict[str, str] | None = None) -> float:
        """Return one counter value for assertions and health rendering."""

        return self._counters.get((name, _tags_key(tags)), 0.0)

    def snapshot(self) -> tuple[MetricSample, ...]:
        """Return a stable snapshot of all counters and observations."""

        counters = tuple(
            MetricSample(name=name, value=value, tags=MappingProxyType(dict(tags)))
            for (name, tags), value in sorted(self._counters.items(), key=lambda item: (item[0][0], item[0][1]))
        )
        return counters + tuple(self._observations)


def _tags_key(tags: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((tags or {}).items()))


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
    "InMemoryMetricsRecorder",
    "MetricSample",
]
