from __future__ import annotations

from decimal import Decimal

from bot_platform_service.domain import BotSignal, BotSignalSide, BotSignalType
from bot_platform_service.trading_bots.spot_grid.domain.models import GridLevelSide, TargetIntent, TargetIntentType
from bot_platform_service.trading_bots.spot_grid.domain.position_intent_contract import (
    POSITION_INTENT_PAYLOAD_SCHEMA,
    POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
)


def target_intent_to_bot_signal(
    intent: TargetIntent,
    *,
    instance_id: str,
    module_id: str,
    snapshot_id: str,
    confidence: Decimal | None,
) -> BotSignal:
    """Convert an execution-neutral Spot Grid target intent into a platform signal."""

    signal_type = _signal_type_from_intent(intent)
    side = _signal_side_from_intent(intent)
    return BotSignal.build(
        instance_id=instance_id,
        module_id=module_id,
        symbol=intent.symbol,
        timeframe=intent.timeframe,
        snapshot_id=snapshot_id,
        signal_type=signal_type,
        side=side,
        confidence=confidence,
        reason=intent.reason_codes[0],
        payload_schema=POSITION_INTENT_PAYLOAD_SCHEMA,
        payload_schema_version=POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
        payload=intent.to_payload(),
    )


def _signal_type_from_intent(intent: TargetIntent) -> BotSignalType:
    if intent.intent_type is TargetIntentType.OPEN_POSITION:
        return BotSignalType.ENTRY
    if intent.intent_type is TargetIntentType.CLOSE_POSITION:
        return BotSignalType.EXIT
    if intent.intent_type is TargetIntentType.REBALANCE:
        return BotSignalType.REBALANCE
    if intent.intent_type is TargetIntentType.HOLD:
        return BotSignalType.HOLD
    return BotSignalType.ALERT


def _signal_side_from_intent(intent: TargetIntent) -> BotSignalSide | None:
    if intent.side is GridLevelSide.BUY:
        return BotSignalSide.BUY
    if intent.side is GridLevelSide.SELL:
        return BotSignalSide.SELL
    return None
