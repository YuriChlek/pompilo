"""Platform-native Spot Greenwich adapter."""

from __future__ import annotations

from decimal import Decimal
from typing import Mapping

from bot_platform_service.domain import (
    BotHealth,
    BotHealthStatus,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMode,
    BotNotificationEvent,
    BotNotificationStatus,
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
from bot_platform_service.trading_bots.spot_greenwich.application import (
    GreenwichTradingCycleService,
    parse_greenwich_config,
)
from bot_platform_service.trading_bots.spot_greenwich.domain import (
    GreenwichActionType,
    GreenwichExecutionDecision,
    GreenwichMultiTimeframePlan,
    GreenwichSignalType,
    GreenwichTradingPlan,
)

MODULE_ID = "spot_greenwich"
CONFIG_SCHEMA_VERSION = 1
SUPPORTED_TIMEFRAMES = {"1d", "4h"}
SUPPORTED_MODES = {BotMode.DRY_RUN, BotMode.NOTIFICATION_ONLY, BotMode.SIGNAL_ONLY}
SIGNAL_PAYLOAD_SCHEMA = "spot_greenwich.execution_decision"
SIGNAL_PAYLOAD_SCHEMA_VERSION = 1


class SpotGreenwichAdapter:
    """BotModule implementation backed by the platform-native Greenwich service."""

    module_id = MODULE_ID

    def __init__(self, cycle_service: GreenwichTradingCycleService | None = None) -> None:
        self._cycle_service = cycle_service or GreenwichTradingCycleService()
        self._context: BotRuntimeContext | None = None
        self._initialized = False

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        errors: list[str] = []
        if config.module_id != self.module_id:
            errors.append("module_id must be spot_greenwich")
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
            return _failed_result(request, "MODULE_MISMATCH", "request module_id must be spot_greenwich")
        if request.mode not in SUPPORTED_MODES:
            return _failed_result(request, "MODE_UNSUPPORTED", "request mode is unsupported")
        if request.market_data is None:
            return _failed_result(request, "MARKET_DATA_REQUIRED", "primary market snapshot is required")

        try:
            config = _config_from_request(request)
            cycle = self._cycle_service.run_once(market_data=request.market_data, config=config)
            plan = cycle.plan
            signal = _signal_from_plan(request, plan)
            notifications = _notifications_from_plan(request, plan) if request.mode is BotMode.NOTIFICATION_ONLY else ()
            return BotRunResult(
                run_id=request.run_id,
                instance_id=request.instance_id,
                module_id=request.module_id,
                mode=request.mode,
                status=BotRunStatus.COMPLETE,
                signals=(signal,),
                notifications=notifications,
                diagnostics=_diagnostics(cycle.diagnostics, request=request, initialized=self._initialized, signal_count=1),
                state_changes=(_state_change(request, plan, request.market_data),),
            )
        except (ValueError, TypeError, ArithmeticError) as exc:
            return _failed_result(
                request,
                "SPOT_GREENWICH_RUN_FAILED",
                f"spot_greenwich run failed: {type(exc).__name__}",
            )


def _validate_module_config(
    payload: Mapping[str, object],
    *,
    symbols: tuple[str, ...],
    timeframes: tuple[str, ...],
) -> list[str]:
    errors: list[str] = []
    try:
        parsed = parse_greenwich_config(payload, fallback_symbols=symbols, fallback_timeframes=timeframes)
    except (ValueError, TypeError, ArithmeticError) as exc:
        return [f"config is invalid: {type(exc).__name__}"]
    if not parsed.symbols:
        errors.append("config symbols must not be empty")
    if parsed.primary_timeframe not in SUPPORTED_TIMEFRAMES:
        errors.append(f"config primary_timeframe is unsupported: {parsed.primary_timeframe}")
    unsupported_supporting = sorted(set(parsed.supporting_timeframes) - SUPPORTED_TIMEFRAMES)
    if unsupported_supporting:
        errors.append(f"config supporting_timeframes contain unsupported values: {', '.join(unsupported_supporting)}")
    if parsed.signal.length < 1:
        errors.append("config greenwich_length must be positive")
    if parsed.execution.deposit_percent <= Decimal("0"):
        errors.append("config deposit_percent must be positive")
    if parsed.execution.deposit_percent > Decimal("100"):
        errors.append("config deposit_percent must be less than or equal to 100")
    if parsed.execution.min_profit_ratio < Decimal("0"):
        errors.append("config min_profit_ratio must not be negative")
    return errors


def _config_from_request(request: BotRunRequest):
    market_data = request.market_data
    if market_data is None:
        raise ValueError("market_data is required")
    primary = market_data.primary_snapshot
    supporting_timeframes = tuple(snapshot.timeframe for snapshot in market_data.supporting_snapshots)
    fallback_timeframes = (primary.timeframe, *supporting_timeframes)
    return parse_greenwich_config(
        request.config,
        fallback_symbols=(primary.canonical_symbol,),
        fallback_timeframes=fallback_timeframes,
    )


def _signal_from_plan(request: BotRunRequest, plan: GreenwichTradingPlan | GreenwichMultiTimeframePlan) -> BotSignal:
    decision = plan.decision
    signal = plan.signal.resolved if isinstance(plan, GreenwichMultiTimeframePlan) else plan.signal
    signal_type = _bot_signal_type(decision)
    side = _bot_signal_side(decision)
    payload = {
        "action": decision.action.value,
        "strategy_signal_type": signal.signal_type.value,
        "signal_price": str(decision.signal_price),
        "quantity": str(decision.quantity),
        "quote_amount": str(decision.quote_amount),
        "decision_reason": decision.reason,
        "strategy_reason": signal.reason,
        "signal_candle_id": decision.signal_candle_id,
        "planner": plan.diagnostics.get("planner"),
    }
    return BotSignal.build(
        instance_id=request.instance_id,
        module_id=request.module_id,
        symbol=decision.symbol,
        timeframe=decision.signal_timeframe or signal.timeframe,
        snapshot_id=request.market_data.primary_snapshot.snapshot_id if request.market_data is not None else "",
        signal_type=signal_type,
        side=side,
        confidence=_confidence(signal.signal_type),
        reason=decision.reason,
        payload_schema=SIGNAL_PAYLOAD_SCHEMA,
        payload_schema_version=SIGNAL_PAYLOAD_SCHEMA_VERSION,
        payload=payload,
    )


def _bot_signal_type(decision: GreenwichExecutionDecision) -> BotSignalType:
    if decision.action is GreenwichActionType.BUY:
        return BotSignalType.ENTRY
    if decision.action is GreenwichActionType.SELL:
        return BotSignalType.EXIT
    return BotSignalType.HOLD


def _bot_signal_side(decision: GreenwichExecutionDecision) -> BotSignalSide | None:
    if decision.action is GreenwichActionType.BUY:
        return BotSignalSide.BUY
    if decision.action is GreenwichActionType.SELL:
        return BotSignalSide.SELL
    return None


def _confidence(signal_type: GreenwichSignalType) -> Decimal | None:
    if signal_type is GreenwichSignalType.HOLD:
        return Decimal("0.50")
    return Decimal("0.65")


def _notifications_from_plan(
    request: BotRunRequest,
    plan: GreenwichTradingPlan | GreenwichMultiTimeframePlan,
) -> tuple[BotNotificationEvent, ...]:
    decision = plan.decision
    return (
        BotNotificationEvent(
            notification_id=f"{request.run_id}:{request.instance_id}:spot_greenwich:{decision.signal_candle_id or 'latest'}",
            instance_id=request.instance_id,
            module_id=request.module_id,
            status=BotNotificationStatus.SKIPPED,
            channel="platform",
            message_type="spot_greenwich_signal",
        ),
    )


def _state_change(
    request: BotRunRequest,
    plan: GreenwichTradingPlan | GreenwichMultiTimeframePlan,
    market_data: BotMarketDataContext,
) -> BotStateChange:
    snapshot = market_data.primary_snapshot
    decision = plan.decision
    signal = plan.signal.resolved if isinstance(plan, GreenwichMultiTimeframePlan) else plan.signal
    return BotStateChange(
        instance_id=request.instance_id,
        namespace="spot_greenwich",
        state_key=f"{decision.symbol}:{decision.signal_timeframe or signal.timeframe}:last_decision",
        operation=BotStateChangeOperation.UPSERT,
        value={
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_version": snapshot.snapshot_version,
            "data_hash": snapshot.data_hash,
            "symbol": decision.symbol,
            "timeframe": decision.signal_timeframe or signal.timeframe,
            "action": decision.action.value,
            "strategy_signal_type": signal.signal_type.value,
            "decision_reason": decision.reason,
            "strategy_reason": signal.reason,
            "signal_price": str(decision.signal_price),
            "quantity": str(decision.quantity),
            "quote_amount": str(decision.quote_amount),
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
