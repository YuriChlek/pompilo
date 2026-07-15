"""Platform-native Spot Grid adapter."""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping

from bot_platform_service.domain import (
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMode,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotRuntimeContext,
    BotSignal,
    BotSignalSide,
    BotSignalType,
    BotStartRequest,
    BotStartResult,
    BotStateChange,
    BotStateChangeOperation,
    BotStopResult,
    BotValidationResult,
)
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.domain import GridLevel, GridLevelSide, SpotGridPlan

MODULE_ID = "spot_grid"
CONFIG_SCHEMA_VERSION = 1
SUPPORTED_TIMEFRAMES = {"1h", "4h"}
SUPPORTED_MODES = {BotMode.DRY_RUN, BotMode.NOTIFICATION_ONLY, BotMode.SIGNAL_ONLY}
SIGNAL_PAYLOAD_SCHEMA = "spot_grid.grid_level"
SIGNAL_PAYLOAD_SCHEMA_VERSION = 1


class SpotGridAdapter:
    """BotModule implementation backed by the platform-native Spot Grid service."""

    module_id = MODULE_ID

    def __init__(self, cycle_service: SpotGridTradingCycleService | None = None) -> None:
        self._cycle_service = cycle_service or SpotGridTradingCycleService()
        self._context: BotRuntimeContext | None = None
        self._initialized = False

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        errors: list[str] = []
        if config.module_id != self.module_id:
            errors.append("module_id must be spot_grid")
        if config.config_schema_version != CONFIG_SCHEMA_VERSION:
            errors.append("config_schema_version must be 1")
        if not config.symbols:
            errors.append("symbols must not be empty")
        if not config.timeframes:
            errors.append("timeframes must not be empty")
        unsupported_timeframes = sorted(set(config.timeframes) - SUPPORTED_TIMEFRAMES)
        if unsupported_timeframes:
            errors.append(f"unsupported timeframes: {', '.join(unsupported_timeframes)}")
        if config.mode not in SUPPORTED_MODES:
            errors.append("mode is unsupported")
        if not errors:
            errors.extend(_validate_module_config(config.config, symbols=config.symbols, timeframes=config.timeframes))
        return BotValidationResult(valid=not errors, errors=tuple(errors))

    async def initialize(self, context: BotRuntimeContext) -> None:
        self._context = context
        self._initialized = True

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        return self._run(request)

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        return self._run(request)

    async def start(self, request: BotStartRequest) -> BotStartResult:
        return BotStartResult(
            accepted=False,
            instance_id=request.instance_id,
            error_code="START_NOT_SUPPORTED",
        )

    async def stop(self, instance_id: str) -> BotStopResult:
        return BotStopResult(accepted=True, instance_id=instance_id)

    async def health(self, instance_id: str) -> BotHealth:
        return BotHealth(
            instance_id=instance_id,
            module_id=self.module_id,
            status=BotHealthStatus.HEALTHY if self._initialized else BotHealthStatus.DEGRADED,
            details={
                "initialized": self._initialized,
                "readiness": "ready" if self._initialized else "not_initialized",
            },
        )

    def _run(self, request: BotRunRequest) -> BotRunResult:
        if request.module_id != self.module_id:
            return _failed_result(request, "MODULE_MISMATCH", "request module_id must be spot_grid")
        if request.mode not in SUPPORTED_MODES:
            return _failed_result(request, "MODE_UNSUPPORTED", "request mode is unsupported")
        if request.market_data is None:
            return _failed_result(request, "MARKET_DATA_REQUIRED", "primary market snapshot is required")

        try:
            config = _config_from_market_data(request.market_data)
            cycle = self._cycle_service.run_once(snapshot=request.market_data.primary_snapshot, config=config)
            signals = tuple(_signal_from_level(request, cycle.plan, level) for level in cycle.plan.levels)
            diagnostics = _diagnostics(
                cycle.diagnostics,
                request=request,
                initialized=self._initialized,
                signal_count=len(signals),
            )
            state_changes = (_state_change(request, cycle.plan, request.market_data),)
            return BotRunResult(
                run_id=request.run_id,
                instance_id=request.instance_id,
                module_id=request.module_id,
                mode=request.mode,
                status=BotRunStatus.COMPLETE,
                signals=signals,
                diagnostics=diagnostics,
                state_changes=state_changes,
            )
        except (ValueError, TypeError, ArithmeticError) as exc:
            return _failed_result(
                request,
                "SPOT_GRID_RUN_FAILED",
                f"spot_grid run failed: {type(exc).__name__}",
            )


def _validate_module_config(
    payload: Mapping[str, object],
    *,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    try:
        parsed = parse_spot_grid_config(payload, fallback_symbols=symbols, fallback_timeframes=timeframes)
    except (ValueError, TypeError, ArithmeticError) as exc:
        return [f"config is invalid: {type(exc).__name__}"]
    if not parsed.symbols:
        errors.append("config symbols must not be empty")
    if parsed.primary_timeframe not in SUPPORTED_TIMEFRAMES:
        errors.append(f"config primary_timeframe is unsupported: {parsed.primary_timeframe}")
    unsupported_supporting = sorted(set(parsed.supporting_timeframes) - SUPPORTED_TIMEFRAMES)
    if unsupported_supporting:
        errors.append(f"config supporting_timeframes contain unsupported values: {', '.join(unsupported_supporting)}")
    if parsed.max_position_fraction <= Decimal("0"):
        errors.append("config max_position_fraction must be positive")
    if parsed.max_position_fraction > Decimal("1"):
        errors.append("config max_position_fraction must be less than or equal to 1")
    if parsed.max_grid_levels < 1:
        errors.append("config max_grid_levels must be positive")
    if parsed.max_grid_levels > 50:
        errors.append("config max_grid_levels must be less than or equal to 50")
    return errors


def _config_from_market_data(market_data: BotMarketDataContext):
    primary = market_data.primary_snapshot
    supporting_timeframes = tuple(snapshot.timeframe for snapshot in market_data.supporting_snapshots)
    fallback_timeframes = (primary.timeframe, *supporting_timeframes)
    return parse_spot_grid_config(
        {},
        fallback_symbols=(primary.canonical_symbol,),
        fallback_timeframes=fallback_timeframes,
    )


def _signal_from_level(request: BotRunRequest, plan: SpotGridPlan, level: GridLevel) -> BotSignal:
    signal_type = BotSignalType.ENTRY if level.side is GridLevelSide.BUY else BotSignalType.EXIT
    side = BotSignalSide.BUY if level.side is GridLevelSide.BUY else BotSignalSide.SELL
    payload = {
        "level_index": level.level_index,
        "level_side": level.side.value,
        "grid_price": str(level.price),
        "reference_price": str(plan.reference_price),
        "range_low": str(plan.range_low),
        "range_high": str(plan.range_high),
        "planner": plan.diagnostics.get("planner"),
    }
    return BotSignal.build(
        instance_id=request.instance_id,
        module_id=request.module_id,
        symbol=plan.symbol,
        timeframe=plan.timeframe,
        snapshot_id=request.market_data.primary_snapshot.snapshot_id if request.market_data is not None else "",
        signal_type=signal_type,
        side=side,
        confidence=Decimal("0.60"),
        reason=level.reason,
        payload_schema=SIGNAL_PAYLOAD_SCHEMA,
        payload_schema_version=SIGNAL_PAYLOAD_SCHEMA_VERSION,
        payload=payload,
    )


def _state_change(request: BotRunRequest, plan: SpotGridPlan, market_data: BotMarketDataContext) -> BotStateChange:
    snapshot = market_data.primary_snapshot
    return BotStateChange(
        instance_id=request.instance_id,
        namespace="spot_grid",
        state_key=f"{plan.symbol}:{plan.timeframe}:last_plan",
        operation=BotStateChangeOperation.UPSERT,
        value={
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_version": snapshot.snapshot_version,
            "data_hash": snapshot.data_hash,
            "symbol": plan.symbol,
            "timeframe": plan.timeframe,
            "reference_price": str(plan.reference_price),
            "range_low": str(plan.range_low),
            "range_high": str(plan.range_high),
            "level_count": len(plan.levels),
            "mode": request.mode.value,
        },
    )


def _diagnostics(
    cycle_diagnostics: Mapping[str, object],
    *,
    request: BotRunRequest,
    initialized: bool,
    signal_count: int,
) -> dict[str, object]:
    return {
        **cycle_diagnostics,
        "initialized": initialized,
        "mode": request.mode.value,
        "signal_count": signal_count,
        "readiness": "ready",
    }


def _failed_result(request: BotRunRequest, error_code: str, redacted_message: str) -> BotRunResult:
    return BotRunResult(
        run_id=request.run_id,
        instance_id=request.instance_id,
        module_id=request.module_id,
        mode=request.mode,
        status=BotRunStatus.FAILED,
        diagnostics={
            "readiness": "failed",
        },
        error_code=error_code,
        error_message_redacted=redacted_message,
    )
