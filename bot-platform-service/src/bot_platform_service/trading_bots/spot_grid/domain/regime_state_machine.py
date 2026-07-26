from __future__ import annotations

from typing import Mapping

from bot_platform_service.trading_bots.spot_grid.domain.models import (
    MarketRegime,
    RegimeSnapshot,
    RegimeStateSnapshot,
    normalize_spot_grid_symbol,
)

REGIME_TRANSITION_CONFIRMATIONS = 2
COOLDOWN_REGIMES = frozenset(
    (
        MarketRegime.RANGE,
        MarketRegime.UPTREND,
        MarketRegime.DOWNTREND,
        MarketRegime.HIGH_VOLATILITY,
    )
)


def resolve_regime_state(
    *,
    symbol: str,
    timeframe: str,
    detected: RegimeSnapshot,
    snapshot_id: str,
    snapshot_version: int,
    data_hash: str,
    previous_state: object | None,
) -> RegimeStateSnapshot:
    """Resolve effective regime using persisted state, hysteresis, and cooldown."""

    normalized_symbol = normalize_spot_grid_symbol(symbol)
    previous_payload = _previous_payload(previous_state)
    previous_regime = _previous_regime(previous_payload)
    previous_pending = _previous_pending_regime(previous_payload)
    previous_pending_count = _previous_pending_count(previous_payload)
    reasons: list[str] = []

    if previous_state is not None and previous_payload is None:
        reasons.append("previous_state_ignored")

    if previous_payload is not None and previous_payload.get("snapshot_id") == snapshot_id:
        reasons.append("repeated_snapshot_state_preserved")
        return RegimeStateSnapshot(
            symbol=normalized_symbol,
            timeframe=timeframe,
            effective_regime=previous_regime or detected.regime,
            detected_regime=detected.regime,
            previous_regime=previous_regime,
            pending_regime=previous_pending,
            pending_confirmation_count=previous_pending_count,
            transition_accepted=False,
            snapshot_id=snapshot_id,
            snapshot_version=snapshot_version,
            data_hash=data_hash,
            reasons=tuple(reasons),
            diagnostics=_diagnostics(detected=detected, previous_payload=previous_payload),
        )

    if previous_regime is None:
        reasons.append("initial_regime_state")
        return RegimeStateSnapshot(
            symbol=normalized_symbol,
            timeframe=timeframe,
            effective_regime=detected.regime,
            detected_regime=detected.regime,
            previous_regime=None,
            pending_regime=None,
            pending_confirmation_count=0,
            transition_accepted=True,
            snapshot_id=snapshot_id,
            snapshot_version=snapshot_version,
            data_hash=data_hash,
            reasons=tuple(reasons),
            diagnostics=_diagnostics(detected=detected, previous_payload=previous_payload),
        )

    if detected.regime is previous_regime:
        reasons.append("regime_unchanged")
        return RegimeStateSnapshot(
            symbol=normalized_symbol,
            timeframe=timeframe,
            effective_regime=previous_regime,
            detected_regime=detected.regime,
            previous_regime=previous_regime,
            pending_regime=None,
            pending_confirmation_count=0,
            transition_accepted=False,
            snapshot_id=snapshot_id,
            snapshot_version=snapshot_version,
            data_hash=data_hash,
            reasons=tuple(reasons),
            diagnostics=_diagnostics(detected=detected, previous_payload=previous_payload),
        )

    if _requires_confirmation(previous_regime, detected.regime):
        pending_count = previous_pending_count + 1 if previous_pending is detected.regime else 1
        if pending_count < REGIME_TRANSITION_CONFIRMATIONS:
            reasons.append("regime_transition_pending_confirmation")
            return RegimeStateSnapshot(
                symbol=normalized_symbol,
                timeframe=timeframe,
                effective_regime=previous_regime,
                detected_regime=detected.regime,
                previous_regime=previous_regime,
                pending_regime=detected.regime,
                pending_confirmation_count=pending_count,
                transition_accepted=False,
                snapshot_id=snapshot_id,
                snapshot_version=snapshot_version,
                data_hash=data_hash,
                reasons=tuple(reasons),
                diagnostics=_diagnostics(detected=detected, previous_payload=previous_payload),
            )
        reasons.append("regime_transition_confirmed")

    else:
        reasons.append("regime_transition_immediate")

    return RegimeStateSnapshot(
        symbol=normalized_symbol,
        timeframe=timeframe,
        effective_regime=detected.regime,
        detected_regime=detected.regime,
        previous_regime=previous_regime,
        pending_regime=None,
        pending_confirmation_count=0,
        transition_accepted=True,
        snapshot_id=snapshot_id,
        snapshot_version=snapshot_version,
        data_hash=data_hash,
        reasons=tuple(reasons),
        diagnostics=_diagnostics(detected=detected, previous_payload=previous_payload),
    )


def regime_state_key(*, symbol: str, timeframe: str) -> str:
    return f"{normalize_spot_grid_symbol(symbol)}:{timeframe}:regime_state"


def _requires_confirmation(previous: MarketRegime, detected: MarketRegime) -> bool:
    return previous in COOLDOWN_REGIMES and detected in COOLDOWN_REGIMES


def _previous_payload(value: object | None) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _previous_regime(payload: Mapping[str, object] | None) -> MarketRegime | None:
    if payload is None:
        return None
    raw = payload.get("effective_regime")
    try:
        return MarketRegime(str(raw))
    except ValueError:
        return None


def _previous_pending_regime(payload: Mapping[str, object] | None) -> MarketRegime | None:
    if payload is None or payload.get("pending_regime") is None:
        return None
    try:
        return MarketRegime(str(payload.get("pending_regime")))
    except ValueError:
        return None


def _previous_pending_count(payload: Mapping[str, object] | None) -> int:
    if payload is None:
        return 0
    value = payload.get("pending_confirmation_count")
    return value if isinstance(value, int) and value > 0 else 0


def _diagnostics(
    *,
    detected: RegimeSnapshot,
    previous_payload: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "detected_confidence": detected.confidence,
        "detected_reasons": detected.reasons,
        "previous_snapshot_id": previous_payload.get("snapshot_id") if previous_payload is not None else None,
        "transition_confirmations_required": REGIME_TRANSITION_CONFIRMATIONS,
    }


__all__ = [
    "COOLDOWN_REGIMES",
    "REGIME_TRANSITION_CONFIRMATIONS",
    "regime_state_key",
    "resolve_regime_state",
]
