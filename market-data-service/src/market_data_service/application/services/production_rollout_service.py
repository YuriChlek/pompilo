from __future__ import annotations

from market_data_service.domain.rollout_models import (
    RolloutDecision,
    RolloutDecisionStatus,
    RolloutHealthSnapshot,
    RolloutScope,
)


DEFAULT_REQUIRED_STABLE_MINUTES = 60
DEFAULT_MAX_OUTBOX_LAG_SECONDS = 300.0


def build_symbol_group_rollout_scopes(
    *,
    provider_symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
    symbol_group_size: int = 1,
) -> tuple[RolloutScope, ...]:
    if symbol_group_size <= 0:
        raise ValueError("symbol_group_size must be positive")
    normalized_symbols = tuple(symbol.strip().upper() for symbol in provider_symbols if symbol.strip())
    normalized_timeframes = tuple(timeframe.strip().lower() for timeframe in timeframes if timeframe.strip())
    scopes: list[RolloutScope] = []
    for index in range(0, len(normalized_symbols), symbol_group_size):
        symbols = normalized_symbols[index : index + symbol_group_size]
        scopes.append(
            RolloutScope(
                name=f"symbol-group-{len(scopes) + 1}",
                provider_symbols=symbols,
                timeframes=normalized_timeframes,
            )
        )
    return tuple(scopes)


def build_timeframe_group_rollout_scopes(
    *,
    provider_symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
) -> tuple[RolloutScope, ...]:
    normalized_symbols = tuple(symbol.strip().upper() for symbol in provider_symbols if symbol.strip())
    return tuple(
        RolloutScope(
            name=f"timeframe-{timeframe.strip().lower()}",
            provider_symbols=normalized_symbols,
            timeframes=(timeframe.strip().lower(),),
        )
        for timeframe in timeframes
        if timeframe.strip()
    )


def evaluate_rollout_readiness(
    snapshot: RolloutHealthSnapshot,
    *,
    required_stable_minutes: int = DEFAULT_REQUIRED_STABLE_MINUTES,
    max_outbox_lag_seconds: float = DEFAULT_MAX_OUTBOX_LAG_SECONDS,
) -> RolloutDecision:
    reasons: list[str] = []
    rollback_reasons: list[str] = []

    if snapshot.candles_created <= 0:
        rollback_reasons.append("no candles created")
    if snapshot.complete_batches <= 0:
        rollback_reasons.append("no complete batches")
    if snapshot.snapshots_created <= 0:
        rollback_reasons.append("no snapshots created")
    if snapshot.outbox_events_created <= 0:
        rollback_reasons.append("no outbox events created")

    critical_alerts = tuple(alert for alert in snapshot.active_alert_names if alert.endswith("_critical"))
    if critical_alerts:
        rollback_reasons.append(f"critical alerts active: {', '.join(critical_alerts)}")

    if rollback_reasons:
        return RolloutDecision(
            status=RolloutDecisionStatus.ROLLBACK,
            reasons=tuple(rollback_reasons),
            ready_for_consumer_integration_plan=False,
        )

    if snapshot.active_alert_names:
        reasons.append(f"alerts active: {', '.join(snapshot.active_alert_names)}")
    if snapshot.outbox_lag_seconds > max_outbox_lag_seconds:
        reasons.append("outbox lag above threshold")
    if snapshot.stable_minutes < required_stable_minutes:
        reasons.append("stable period is not complete")

    if reasons:
        return RolloutDecision(
            status=RolloutDecisionStatus.HOLD,
            reasons=tuple(reasons),
            ready_for_consumer_integration_plan=False,
        )

    return RolloutDecision(
        status=RolloutDecisionStatus.PROMOTE,
        reasons=("stable ingestion rollout scope is ready",),
        ready_for_consumer_integration_plan=True,
    )
