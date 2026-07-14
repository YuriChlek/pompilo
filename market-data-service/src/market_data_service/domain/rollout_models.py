from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RolloutDecisionStatus(StrEnum):
    HOLD = "HOLD"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True, slots=True)
class RolloutScope:
    name: str
    provider_symbols: tuple[str, ...]
    timeframes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RolloutHealthSnapshot:
    candles_created: int
    complete_batches: int
    snapshots_created: int
    outbox_events_created: int
    outbox_lag_seconds: float
    active_alert_names: tuple[str, ...]
    stable_minutes: int


@dataclass(frozen=True, slots=True)
class RolloutDecision:
    status: RolloutDecisionStatus
    reasons: tuple[str, ...]
    ready_for_consumer_integration_plan: bool
