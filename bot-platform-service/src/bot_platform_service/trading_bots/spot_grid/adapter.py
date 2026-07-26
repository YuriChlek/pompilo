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
    BotNotificationEvent,
    BotNotificationStatus,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotRuntimeContext,
    BotSignal,
    BotStartRequest,
    BotStartResult,
    BotStateChange,
    BotStateChangeOperation,
    BotStopResult,
    BotValidationResult,
)
from bot_platform_service.trading_bots.spot_grid.application import (
    SpotGridPortfolioContextProvider,
    SpotGridTradingCycleService,
    parse_spot_grid_config,
)
from bot_platform_service.trading_bots.spot_grid.domain import (
    POSITION_INTENT_PAYLOAD_SCHEMA,
    POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
    PortfolioContext,
    SpotGridPlan,
    TargetIntent,
    build_semantic_intent_hash,
    normalize_spot_grid_symbol,
    regime_state_key,
    target_intent_to_bot_signal,
)

MODULE_ID = "spot_grid"
CONFIG_SCHEMA_VERSION = 1
SUPPORTED_TIMEFRAMES = {"1h", "4h"}
SUPPORTED_MODES = {BotMode.DRY_RUN, BotMode.NOTIFICATION_ONLY, BotMode.SIGNAL_ONLY}
SIGNAL_PAYLOAD_SCHEMA = POSITION_INTENT_PAYLOAD_SCHEMA
SIGNAL_PAYLOAD_SCHEMA_VERSION = POSITION_INTENT_PAYLOAD_SCHEMA_VERSION


class SpotGridAdapter:
    """BotModule implementation backed by the platform-native Spot Grid service."""

    module_id = MODULE_ID

    def __init__(
        self,
        cycle_service: SpotGridTradingCycleService | None = None,
        portfolio_context_provider: SpotGridPortfolioContextProvider | None = None,
    ) -> None:
        self._cycle_service = cycle_service or SpotGridTradingCycleService()
        self._portfolio_context_provider = portfolio_context_provider
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
        return await self._run(request)

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        return await self._run(request)

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

    async def _run(self, request: BotRunRequest) -> BotRunResult:
        if request.module_id != self.module_id:
            return _failed_result(request, "MODULE_MISMATCH", "request module_id must be spot_grid")
        if request.mode not in SUPPORTED_MODES:
            return _failed_result(request, "MODE_UNSUPPORTED", "request mode is unsupported")
        if request.market_data is None:
            return _failed_result(request, "MARKET_DATA_REQUIRED", "primary market snapshot is required")

        try:
            config = _config_from_request(request)
            previous_regime_state = await self._load_previous_regime_state(request)
            previous_cooldown_state = await self._load_previous_cooldown_state(request)
            previous_intent_state = await self._load_previous_intent_state(request)
            portfolio_context = await self._load_portfolio_context(request)
            cycle = self._cycle_service.run_once(
                market_data=request.market_data,
                config=config,
                previous_regime_state=previous_regime_state,
                previous_cooldown_state=previous_cooldown_state,
                portfolio_context=portfolio_context,
            )
            dedupe = _dedupe_intents(cycle.plan.intents, previous_intent_state=previous_intent_state)
            signals = tuple(_signal_from_intent(request, intent) for intent in dedupe.intents_to_emit)
            diagnostics = _diagnostics(
                {**cycle.diagnostics, "semantic_intent_idempotency": dedupe.diagnostics},
                request=request,
                initialized=self._initialized,
                signal_count=len(signals),
            )
            state_changes = _state_changes(request, cycle.plan, request.market_data, cycle.diagnostics, dedupe)
            return BotRunResult(
                run_id=request.run_id,
                instance_id=request.instance_id,
                module_id=request.module_id,
                mode=request.mode,
                status=BotRunStatus.COMPLETE,
                signals=signals,
                notifications=_notifications_from_signals(request, signals)
                if request.mode is BotMode.NOTIFICATION_ONLY
                else (),
                diagnostics=diagnostics,
                state_changes=state_changes,
            )
        except (ValueError, TypeError, ArithmeticError) as exc:
            return _failed_result(
                request,
                "SPOT_GRID_RUN_FAILED",
                f"spot_grid run failed: {type(exc).__name__}",
            )

    async def _load_previous_regime_state(self, request: BotRunRequest) -> object | None:
        if self._context is None:
            return None
        if request.market_data is None:
            return None
        snapshot = request.market_data.primary_snapshot
        state_key = regime_state_key(symbol=snapshot.canonical_symbol, timeframe=snapshot.timeframe)
        return await self._context.state_store.load(
            instance_id=request.instance_id,
            namespace="spot_grid",
            state_key=state_key,
        )

    async def _load_previous_cooldown_state(self, request: BotRunRequest) -> object | None:
        if self._context is None:
            return None
        if request.market_data is None:
            return None
        snapshot = request.market_data.primary_snapshot
        return await self._context.state_store.load(
            instance_id=request.instance_id,
            namespace="spot_grid",
            state_key=_cooldown_state_key(symbol=snapshot.canonical_symbol, timeframe=snapshot.timeframe),
        )

    async def _load_previous_intent_state(self, request: BotRunRequest) -> object | None:
        if self._context is None:
            return None
        if request.market_data is None:
            return None
        snapshot = request.market_data.primary_snapshot
        return await self._context.state_store.load(
            instance_id=request.instance_id,
            namespace="spot_grid",
            state_key=_intent_hash_state_key(symbol=snapshot.canonical_symbol, timeframe=snapshot.timeframe),
        )

    async def _load_portfolio_context(self, request: BotRunRequest) -> PortfolioContext | None:
        if self._portfolio_context_provider is None:
            return None
        if request.market_data is None:
            return None
        snapshot = request.market_data.primary_snapshot
        return await self._portfolio_context_provider.get_portfolio_context(
            instance_id=request.instance_id,
            symbol=snapshot.canonical_symbol,
            timeframe=snapshot.timeframe,
            snapshot_id=snapshot.snapshot_id,
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
    if parsed.min_price_distance_fraction < Decimal("0"):
        errors.append("config min_price_distance_fraction must be non-negative")
    if parsed.min_price_distance_fraction > Decimal("0.10"):
        errors.append("config min_price_distance_fraction must be less than or equal to 0.10")
    if parsed.high_volatility_pause_threshold <= Decimal("0"):
        errors.append("config high_volatility_pause_threshold must be positive")
    if parsed.volatility_cooldown_runs < 0:
        errors.append("config volatility_cooldown_runs must be non-negative")
    return errors


def _config_from_request(request: BotRunRequest):
    market_data = request.market_data
    if market_data is None:
        raise ValueError("market_data is required")
    primary = market_data.primary_snapshot
    supporting_timeframes = tuple(snapshot.timeframe for snapshot in market_data.supporting_snapshots)
    fallback_timeframes = (primary.timeframe, *supporting_timeframes)
    return parse_spot_grid_config(
        request.config,
        fallback_symbols=(primary.canonical_symbol,),
        fallback_timeframes=fallback_timeframes,
    )


def _signal_from_intent(request: BotRunRequest, intent: TargetIntent) -> BotSignal:
    return target_intent_to_bot_signal(
        intent,
        instance_id=request.instance_id,
        module_id=request.module_id,
        snapshot_id=request.market_data.primary_snapshot.snapshot_id if request.market_data is not None else "",
        confidence=Decimal("0.60"),
    )


def _notifications_from_signals(
    request: BotRunRequest,
    signals: tuple[BotSignal, ...],
) -> tuple[BotNotificationEvent, ...]:
    return tuple(
        BotNotificationEvent(
            notification_id=f"{request.run_id}:{request.instance_id}:spot_grid:{signal.signal_key[:16]}",
            instance_id=request.instance_id,
            module_id=request.module_id,
            status=BotNotificationStatus.SKIPPED,
            channel="platform",
            message_type="spot_grid_signal",
        )
        for signal in signals
    )


def _state_changes(
    request: BotRunRequest,
    plan: SpotGridPlan,
    market_data: BotMarketDataContext,
    diagnostics: Mapping[str, object],
    intent_dedupe: "_IntentDedupeResult",
) -> tuple[BotStateChange, ...]:
    regime_state = diagnostics.get("regime_state")
    cooldown_state = diagnostics.get("cooldown_state")
    changes = [_last_plan_state_change(request, plan, market_data)]
    if isinstance(regime_state, Mapping):
        changes.append(_regime_state_change(request, plan, regime_state))
    if isinstance(cooldown_state, Mapping):
        changes.append(_cooldown_state_change(request, plan, cooldown_state))
    if intent_dedupe.current_intent_hashes:
        changes.append(_intent_hash_state_change(request, plan, market_data, intent_dedupe))
    return tuple(changes)


def _last_plan_state_change(request: BotRunRequest, plan: SpotGridPlan, market_data: BotMarketDataContext) -> BotStateChange:
    snapshot = market_data.primary_snapshot
    return BotStateChange(
        instance_id=request.instance_id,
        namespace="spot_grid",
        state_key=_last_plan_state_key(symbol=plan.symbol, timeframe=plan.timeframe),
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
            "regime": plan.regime.value,
            "supporting_snapshots": [
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "snapshot_version": snapshot.snapshot_version,
                    "timeframe": snapshot.timeframe,
                    "data_hash": snapshot.data_hash,
                }
                for snapshot in market_data.supporting_snapshots
            ],
            "mode": request.mode.value,
        },
    )


def _regime_state_change(
    request: BotRunRequest,
    plan: SpotGridPlan,
    regime_state: Mapping[str, object],
) -> BotStateChange:
    return BotStateChange(
        instance_id=request.instance_id,
        namespace="spot_grid",
        state_key=regime_state_key(symbol=plan.symbol, timeframe=plan.timeframe),
        operation=BotStateChangeOperation.UPSERT,
        value=dict(regime_state),
    )


def _cooldown_state_change(
    request: BotRunRequest,
    plan: SpotGridPlan,
    cooldown_state: Mapping[str, object],
) -> BotStateChange:
    return BotStateChange(
        instance_id=request.instance_id,
        namespace="spot_grid",
        state_key=_cooldown_state_key(symbol=plan.symbol, timeframe=plan.timeframe),
        operation=BotStateChangeOperation.UPSERT,
        value=dict(cooldown_state),
    )


def _intent_hash_state_change(
    request: BotRunRequest,
    plan: SpotGridPlan,
    market_data: BotMarketDataContext,
    intent_dedupe: "_IntentDedupeResult",
) -> BotStateChange:
    snapshot = market_data.primary_snapshot
    return BotStateChange(
        instance_id=request.instance_id,
        namespace="spot_grid",
        state_key=_intent_hash_state_key(symbol=plan.symbol, timeframe=plan.timeframe),
        operation=BotStateChangeOperation.UPSERT,
        value={
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_version": snapshot.snapshot_version,
            "data_hash": snapshot.data_hash,
            "symbol": plan.symbol,
            "timeframe": plan.timeframe,
            "intent_hashes": list(intent_dedupe.current_intent_hashes),
            "emitted_intent_hashes": list(intent_dedupe.emitted_intent_hashes),
            "blocked_intent_hashes": list(intent_dedupe.blocked_intent_hashes),
        },
    )


def _cooldown_state_key(*, symbol: str, timeframe: str) -> str:
    return f"{normalize_spot_grid_symbol(symbol)}:{timeframe}:cooldown_state"


def _last_plan_state_key(*, symbol: str, timeframe: str) -> str:
    return f"{normalize_spot_grid_symbol(symbol)}:{timeframe}:last_plan"


def _intent_hash_state_key(*, symbol: str, timeframe: str) -> str:
    return f"{normalize_spot_grid_symbol(symbol)}:{timeframe}:last_emitted_intent_hash"


class _IntentDedupeResult:
    def __init__(
        self,
        *,
        intents_to_emit: tuple[TargetIntent, ...],
        current_intent_hashes: tuple[str, ...],
        emitted_intent_hashes: tuple[str, ...],
        blocked_intent_hashes: tuple[str, ...],
        previous_intent_hashes: tuple[str, ...],
    ) -> None:
        self.intents_to_emit = intents_to_emit
        self.current_intent_hashes = current_intent_hashes
        self.emitted_intent_hashes = emitted_intent_hashes
        self.blocked_intent_hashes = blocked_intent_hashes
        self.previous_intent_hashes = previous_intent_hashes

    @property
    def diagnostics(self) -> dict[str, object]:
        return {
            "current_intent_hashes": self.current_intent_hashes,
            "emitted_intent_hashes": self.emitted_intent_hashes,
            "blocked_intent_hashes": self.blocked_intent_hashes,
            "previous_intent_hashes": self.previous_intent_hashes,
            "blocked_count": len(self.blocked_intent_hashes),
            "emitted_count": len(self.emitted_intent_hashes),
        }


def _dedupe_intents(
    intents: tuple[TargetIntent, ...],
    *,
    previous_intent_state: object | None,
) -> _IntentDedupeResult:
    previous_hashes = _previous_intent_hashes(previous_intent_state)
    previous_hash_set = set(previous_hashes)
    intents_to_emit: list[TargetIntent] = []
    current_hashes: list[str] = []
    emitted_hashes: list[str] = []
    blocked_hashes: list[str] = []

    for intent in intents:
        intent_hash = build_semantic_intent_hash(intent)
        current_hashes.append(intent_hash)
        if intent_hash in previous_hash_set:
            blocked_hashes.append(intent_hash)
            continue
        intents_to_emit.append(intent)
        emitted_hashes.append(intent_hash)

    return _IntentDedupeResult(
        intents_to_emit=tuple(intents_to_emit),
        current_intent_hashes=tuple(current_hashes),
        emitted_intent_hashes=tuple(emitted_hashes),
        blocked_intent_hashes=tuple(blocked_hashes),
        previous_intent_hashes=previous_hashes,
    )


def _previous_intent_hashes(previous_intent_state: object | None) -> tuple[str, ...]:
    if not isinstance(previous_intent_state, Mapping):
        return ()
    raw_hashes = previous_intent_state.get("intent_hashes")
    if isinstance(raw_hashes, str):
        return (raw_hashes,)
    if isinstance(raw_hashes, tuple | list):
        return tuple(value for value in (str(item) for item in raw_hashes) if len(value) == 64)
    raw_hash = previous_intent_state.get("intent_hash")
    if isinstance(raw_hash, str) and len(raw_hash) == 64:
        return (raw_hash,)
    return ()


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
