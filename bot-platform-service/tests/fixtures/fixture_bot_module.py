from __future__ import annotations

from decimal import Decimal

from bot_platform_service.domain import (
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotRuntimeContext,
    BotSignal,
    BotSignalSide,
    BotSignalType,
    BotStartRequest,
    BotStartResult,
    BotStopResult,
    BotValidationResult,
)


class FixtureBotModule:
    """Minimal conforming bot module used by contract harness tests."""

    module_id = "fixture"

    def __init__(self) -> None:
        self.initialized = False

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        return BotValidationResult(valid=config.module_id == self.module_id)

    async def initialize(self, context: BotRuntimeContext) -> None:
        self.initialized = True

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        return BotRunResult(
            run_id=request.run_id,
            instance_id=request.instance_id,
            module_id=request.module_id,
            mode=request.mode,
            status=BotRunStatus.COMPLETE,
            signals=(_signal(request),),
            diagnostics={"fixture": True},
        )

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        return await self.dry_run(request)

    async def start(self, request: BotStartRequest) -> BotStartResult:
        return BotStartResult(accepted=True, instance_id=request.instance_id)

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(instance_id=instance_id, module_id=self.module_id, status=BotHealthStatus.HEALTHY)


class BadRunResultBotModule(FixtureBotModule):
    """Non-conforming module used to prove the harness rejects bad modules."""

    async def dry_run(self, request: BotRunRequest):  # noqa: ANN201 - intentionally invalid contract
        return {"run_id": request.run_id}


def _signal(request: BotRunRequest) -> BotSignal:
    snapshot_id = request.market_data.primary_snapshot.snapshot_id if request.market_data is not None else "snapshot-1"
    timeframe = request.market_data.primary_snapshot.timeframe if request.market_data is not None else "1h"
    return BotSignal.build(
        instance_id=request.instance_id,
        module_id=request.module_id,
        symbol="ETHUSDT",
        timeframe=timeframe,
        snapshot_id=snapshot_id,
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.8"),
        reason="fixture_entry",
        payload_schema="fixture.entry",
        payload_schema_version=1,
        payload={"price": Decimal("100")},
    )
