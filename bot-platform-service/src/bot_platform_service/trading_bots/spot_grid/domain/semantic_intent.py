from __future__ import annotations

from bot_platform_service.domain import build_payload_hash
from bot_platform_service.trading_bots.spot_grid.domain.models import TargetIntent


def build_semantic_intent_hash(intent: TargetIntent) -> str:
    """Build a cross-snapshot idempotency hash for one material strategy intent."""

    return build_payload_hash(semantic_intent_payload(intent))


def semantic_intent_payload(intent: TargetIntent) -> dict[str, object]:
    """Return the stable intent shape used for semantic dedupe state."""

    payload: dict[str, object] = {
        "intent_type": intent.intent_type.value,
        "execution_intent": intent.execution_intent.value,
        "strategy": "spot_grid",
        "regime": intent.regime.value,
        "symbol": intent.symbol,
        "timeframe": intent.timeframe,
        "target_price": str(intent.target_price) if intent.target_price is not None else None,
        "grid_level_index": intent.grid_level_index,
        "side": intent.side.value if intent.side is not None else None,
        "price_band": intent.price_band.to_payload(),
        "risk": intent.risk.to_payload(),
        "guards": intent.guards.to_payload(),
        "reason_codes": intent.reason_codes,
    }
    if intent.position is not None:
        payload["position"] = intent.position.to_payload()
    if intent.metadata:
        payload["metadata"] = dict(intent.metadata)
    return payload


__all__ = [
    "build_semantic_intent_hash",
    "semantic_intent_payload",
]
