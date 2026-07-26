from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

JsonPayload = Mapping[str, object]


class GridLevelSide(StrEnum):
    """Side of a planned grid level."""

    BUY = "buy"
    SELL = "sell"


class MarketRegime(StrEnum):
    """Platform-safe market regime classification used by Spot Grid strategy."""

    RANGE = "range"
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    HIGH_VOLATILITY = "high_volatility"
    RISK_OFF = "risk_off"


class MarketStructureBias(StrEnum):
    """Directional bias extracted from recent Spot Grid market structure."""

    RANGE = "range"
    BULLISH = "bullish"
    BEARISH = "bearish"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class TargetIntentType(StrEnum):
    """Execution-neutral strategy intent categories emitted by Spot Grid."""

    OPEN_POSITION = "open_position"
    CLOSE_POSITION = "close_position"
    REBALANCE = "rebalance"
    HOLD = "hold"
    ALERT = "alert"


class TargetExecutionIntent(StrEnum):
    """Execution-facing hint without venue or private execution details."""

    LIMIT_ENTRY_CANDIDATE = "limit_entry_candidate"
    LIMIT_EXIT_CANDIDATE = "limit_exit_candidate"
    REBALANCE_CANDIDATE = "rebalance_candidate"
    DE_RISK_CANDIDATE = "de_risk_candidate"
    NO_ACTION = "no_action"
    OPERATOR_ALERT = "operator_alert"


_EXECUTION_DETAIL_KEYS = frozenset(
    {
        "_".join(("client", "order", "id")),
        "_".join(("order", "id")),
        "_".join(("exchange", "order", "id")),
        "_".join(("venue", "order", "id")),
        "_".join(("fill", "id")),
        "_".join(("fill", "state")),
        "_".join(("filled", "size")),
        "_".join(("api", "key")),
        "_".join(("api", "secret")),
        "_".join(("private", "key")),
    }
)

_EXECUTION_INTENTS_BY_TYPE = MappingProxyType(
    {
        TargetIntentType.OPEN_POSITION: frozenset(
            {TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE, TargetExecutionIntent.DE_RISK_CANDIDATE}
        ),
        TargetIntentType.CLOSE_POSITION: frozenset(
            {TargetExecutionIntent.LIMIT_EXIT_CANDIDATE, TargetExecutionIntent.DE_RISK_CANDIDATE}
        ),
        TargetIntentType.REBALANCE: frozenset(
            {TargetExecutionIntent.REBALANCE_CANDIDATE, TargetExecutionIntent.DE_RISK_CANDIDATE}
        ),
        TargetIntentType.HOLD: frozenset({TargetExecutionIntent.NO_ACTION}),
        TargetIntentType.ALERT: frozenset({TargetExecutionIntent.OPERATOR_ALERT}),
    }
)


def normalize_spot_grid_symbol(symbol: str) -> str:
    """Normalize symbols to the canonical platform format, for example BTCUSDT."""

    normalized = "".join(character for character in symbol.upper() if character.isalnum())
    if not normalized:
        raise ValueError("symbol must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class SpotGridCandle:
    """Pure candle model used by platform-native Spot Grid planning."""

    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        _require_decimal_fields(
            self,
            ("open", "high", "low", "close", "volume"),
        )


@dataclass(frozen=True, slots=True)
class IndicatorCandle:
    """Library-neutral candle input for indicator adapters."""

    timestamp: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("open", "high", "low", "close", "volume"))

    def to_payload(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "open": _decimal_string(self.open),
            "high": _decimal_string(self.high),
            "low": _decimal_string(self.low),
            "close": _decimal_string(self.close),
            "volume": _decimal_string(self.volume),
        }


@dataclass(frozen=True, slots=True)
class IndicatorInput:
    """Normalized snapshot data prepared for indicator calculation."""

    source: str
    symbol: str
    timeframe: str
    snapshot_id: str
    snapshot_version: int
    data_hash: str
    candles: tuple[IndicatorCandle, ...]
    required_history: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))

    @property
    def candle_count(self) -> int:
        return len(self.candles)

    @property
    def has_required_history(self) -> bool:
        return self.candle_count >= self.required_history

    def to_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "snapshot_id": self.snapshot_id,
            "snapshot_version": self.snapshot_version,
            "data_hash": self.data_hash,
            "candle_count": self.candle_count,
            "required_history": self.required_history,
            "has_required_history": self.has_required_history,
            "candles": [candle.to_payload() for candle in self.candles],
        }


@dataclass(frozen=True, slots=True)
class SwingPoint:
    """Single platform-native swing point extracted from normalized candles."""

    candle_index: int
    timestamp: str
    price: Decimal

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("price",))

    def to_payload(self) -> dict[str, object]:
        return {
            "candle_index": self.candle_index,
            "timestamp": self.timestamp,
            "price": _decimal_string(self.price),
        }


@dataclass(frozen=True, slots=True)
class SupportResistanceCandidate:
    """Local support or resistance level suitable for grid construction."""

    price: Decimal
    source: str
    candle_index: int
    distance_from_close: Decimal

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("price", "distance_from_close"))

    def to_payload(self) -> dict[str, object]:
        return {
            "price": _decimal_string(self.price),
            "source": self.source,
            "candle_index": self.candle_index,
            "distance_from_close": _decimal_string(self.distance_from_close),
        }


@dataclass(frozen=True, slots=True)
class MarketStructureSnapshot:
    """Range, swing, and support/resistance context for Spot Grid planning."""

    bias: MarketStructureBias
    candle_count: int
    has_required_history: bool
    range_low: Decimal | None
    range_high: Decimal | None
    range_position: Decimal | None
    swing_highs: tuple[SwingPoint, ...]
    swing_lows: tuple[SwingPoint, ...]
    support_candidates: tuple[SupportResistanceCandidate, ...]
    resistance_candidates: tuple[SupportResistanceCandidate, ...]
    breakout_direction: str | None
    breakout_reference_price: Decimal | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_decimal_fields(
            self,
            ("range_low", "range_high", "range_position", "breakout_reference_price"),
            allow_none=True,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "bias": self.bias.value,
            "candle_count": self.candle_count,
            "has_required_history": self.has_required_history,
            "range_low": _decimal_string_or_none(self.range_low),
            "range_high": _decimal_string_or_none(self.range_high),
            "range_position": _decimal_string_or_none(self.range_position),
            "swing_highs": [swing.to_payload() for swing in self.swing_highs],
            "swing_lows": [swing.to_payload() for swing in self.swing_lows],
            "support_candidates": [candidate.to_payload() for candidate in self.support_candidates],
            "resistance_candidates": [candidate.to_payload() for candidate in self.resistance_candidates],
            "breakout_direction": self.breakout_direction,
            "breakout_reference_price": _decimal_string_or_none(self.breakout_reference_price),
            "reasons": self.reasons,
        }


@dataclass(frozen=True, slots=True)
class RegimeSnapshot:
    """Single-timeframe Spot Grid market regime decision."""

    regime: MarketRegime
    confidence: Decimal
    reasons: tuple[str, ...]
    diagnostics: JsonPayload = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("confidence",))
        object.__setattr__(self, "diagnostics", MappingProxyType(_json_safe_mapping(self.diagnostics)))

    def to_payload(self) -> dict[str, object]:
        return {
            "regime": self.regime.value,
            "confidence": _decimal_string(self.confidence),
            "reasons": self.reasons,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class RegimeStateSnapshot:
    """Persistable per-symbol/timeframe regime state after hysteresis/cooldown."""

    symbol: str
    timeframe: str
    effective_regime: MarketRegime
    detected_regime: MarketRegime
    previous_regime: MarketRegime | None
    pending_regime: MarketRegime | None
    pending_confirmation_count: int
    transition_accepted: bool
    snapshot_id: str
    snapshot_version: int
    data_hash: str
    reasons: tuple[str, ...]
    diagnostics: JsonPayload = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        object.__setattr__(self, "diagnostics", MappingProxyType(_json_safe_mapping(self.diagnostics)))

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "effective_regime": self.effective_regime.value,
            "detected_regime": self.detected_regime.value,
            "previous_regime": self.previous_regime.value if self.previous_regime is not None else None,
            "pending_regime": self.pending_regime.value if self.pending_regime is not None else None,
            "pending_confirmation_count": self.pending_confirmation_count,
            "transition_accepted": self.transition_accepted,
            "snapshot_id": self.snapshot_id,
            "snapshot_version": self.snapshot_version,
            "data_hash": self.data_hash,
            "reasons": self.reasons,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Core Decimal indicator values used by Spot Grid strategy planning."""

    ema20: Decimal | None
    ema50: Decimal | None
    ema200: Decimal | None
    atr14: Decimal | None
    rsi14: Decimal | None
    realized_volatility: Decimal | None
    realized_volatility_short: Decimal | None
    current_volume: Decimal | None
    volume_ma20: Decimal | None
    volume_ratio: Decimal | None
    candle_count: int
    has_required_history: bool
    volatility_has_required_history: bool
    volume_has_required_history: bool

    def __post_init__(self) -> None:
        _require_decimal_fields(
            self,
            (
                "ema20",
                "ema50",
                "ema200",
                "atr14",
                "rsi14",
                "realized_volatility",
                "realized_volatility_short",
                "current_volume",
                "volume_ma20",
                "volume_ratio",
            ),
            allow_none=True,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "ema20": _decimal_string_or_none(self.ema20),
            "ema50": _decimal_string_or_none(self.ema50),
            "ema200": _decimal_string_or_none(self.ema200),
            "atr14": _decimal_string_or_none(self.atr14),
            "rsi14": _decimal_string_or_none(self.rsi14),
            "realized_volatility": _decimal_string_or_none(self.realized_volatility),
            "realized_volatility_short": _decimal_string_or_none(self.realized_volatility_short),
            "current_volume": _decimal_string_or_none(self.current_volume),
            "volume_ma20": _decimal_string_or_none(self.volume_ma20),
            "volume_ratio": _decimal_string_or_none(self.volume_ratio),
            "candle_count": self.candle_count,
            "has_required_history": self.has_required_history,
            "volatility_has_required_history": self.volatility_has_required_history,
            "volume_has_required_history": self.volume_has_required_history,
        }


@dataclass(frozen=True, slots=True)
class SpotGridConfig:
    """Validated planning settings for one Spot Grid run."""

    symbols: tuple[str, ...]
    primary_timeframe: str
    supporting_timeframes: tuple[str, ...]
    max_position_fraction: Decimal
    max_grid_levels: int
    emit_diagnostics: bool = True
    min_price_distance_fraction: Decimal = Decimal("0.001")
    high_volatility_pause_threshold: Decimal = Decimal("0.05")
    volatility_cooldown_runs: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(normalize_spot_grid_symbol(symbol) for symbol in self.symbols))
        _require_decimal_fields(
            self,
            ("max_position_fraction", "min_price_distance_fraction", "high_volatility_pause_threshold"),
        )


@dataclass(frozen=True, slots=True)
class GridLevel:
    """One planned grid level before conversion to platform signals."""

    side: GridLevelSide
    price: Decimal
    level_index: int
    reason: str

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("price",))


@dataclass(frozen=True, slots=True)
class PriceBand:
    """Strategy price range used to explain one planned target intent."""

    range_low: Decimal
    range_high: Decimal

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("range_low", "range_high"))

    def to_payload(self) -> dict[str, object]:
        return {
            "range_low": str(self.range_low),
            "range_high": str(self.range_high),
        }


@dataclass(frozen=True, slots=True)
class StrategyRiskLimits:
    """Execution-neutral risk hints calculated before venue sizing."""

    max_position_fraction: Decimal
    suggested_quote_notional: Decimal | None = None
    max_quote_notional: Decimal | None = None

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("max_position_fraction",))
        _require_decimal_fields(self, ("suggested_quote_notional", "max_quote_notional"), allow_none=True)

    def to_payload(self) -> dict[str, object]:
        return {
            "max_position_fraction": str(self.max_position_fraction),
            "suggested_quote_notional": _decimal_string_or_none(self.suggested_quote_notional),
            "max_quote_notional": _decimal_string_or_none(self.max_quote_notional),
        }


@dataclass(frozen=True, slots=True)
class StrategyGuardSnapshot:
    """Guardrail values explaining why a strategy intent is allowed or blocked."""

    buy_allowed: bool
    sell_allowed: bool
    rsi14: Decimal | None = None
    atr14: Decimal | None = None
    no_loss_required: bool = False
    no_loss_passed: bool | None = None

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("rsi14", "atr14"), allow_none=True)

    def to_payload(self) -> dict[str, object]:
        return {
            "rsi14": _decimal_string_or_none(self.rsi14),
            "atr14": _decimal_string_or_none(self.atr14),
            "buy_allowed": self.buy_allowed,
            "sell_allowed": self.sell_allowed,
            "no_loss_required": self.no_loss_required,
            "no_loss_passed": self.no_loss_passed,
        }


@dataclass(frozen=True, slots=True)
class PositionContext:
    """Execution-neutral position context supplied to strategy planning."""

    symbol: str
    base_quantity: Decimal = Decimal("0")
    quote_notional: Decimal = Decimal("0")
    cost_basis: Decimal | None = None
    min_no_loss_exit_price: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        _require_decimal_fields(self, ("base_quantity", "quote_notional"))
        _require_decimal_fields(self, ("cost_basis", "min_no_loss_exit_price"), allow_none=True)

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "base_quantity": str(self.base_quantity),
            "quote_notional": str(self.quote_notional),
            "cost_basis": _decimal_string_or_none(self.cost_basis),
            "min_no_loss_exit_price": _decimal_string_or_none(self.min_no_loss_exit_price),
        }


@dataclass(frozen=True, slots=True)
class PortfolioContext:
    """Portfolio-level strategy context without private exchange details."""

    total_equity: Decimal
    available_quote: Decimal
    positions: tuple[PositionContext, ...] = ()

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("total_equity", "available_quote"))

    def to_payload(self) -> dict[str, object]:
        return {
            "total_equity": str(self.total_equity),
            "available_quote": str(self.available_quote),
            "positions": [position.to_payload() for position in self.positions],
        }


@dataclass(frozen=True, slots=True)
class SymbolExposureSnapshot:
    """Deterministic per-symbol exposure derived from platform portfolio input."""

    symbol: str
    position_count: int
    base_quantity: Decimal
    quote_notional: Decimal
    exposure_fraction: Decimal | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        _require_decimal_fields(self, ("base_quantity", "quote_notional", "exposure_fraction"), allow_none=True)

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "position_count": self.position_count,
            "base_quantity": str(self.base_quantity),
            "quote_notional": str(self.quote_notional),
            "exposure_fraction": _decimal_string_or_none(self.exposure_fraction),
        }


@dataclass(frozen=True, slots=True)
class PortfolioExposureSnapshot:
    """Portfolio-level exposure derived without exchange or persistence reads."""

    total_equity: Decimal
    available_quote: Decimal
    gross_quote_notional: Decimal
    gross_exposure_fraction: Decimal | None
    available_quote_fraction: Decimal | None
    per_symbol: tuple[SymbolExposureSnapshot, ...]

    def __post_init__(self) -> None:
        _require_decimal_fields(
            self,
            (
                "total_equity",
                "available_quote",
                "gross_quote_notional",
                "gross_exposure_fraction",
                "available_quote_fraction",
            ),
            allow_none=True,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "portfolio": {
                "total_equity": str(self.total_equity),
                "available_quote": str(self.available_quote),
                "gross_quote_notional": str(self.gross_quote_notional),
                "gross_exposure_fraction": _decimal_string_or_none(self.gross_exposure_fraction),
                "available_quote_fraction": _decimal_string_or_none(self.available_quote_fraction),
            },
            "per_symbol": {
                exposure.symbol: exposure.to_payload()
                for exposure in self.per_symbol
            },
        }


@dataclass(frozen=True, slots=True)
class PortfolioAllocationSnapshot:
    """Execution-neutral budget caps for Spot Grid target intents."""

    symbol: str
    max_position_fraction: Decimal
    symbol_current_quote_notional: Decimal
    symbol_max_quote_notional: Decimal
    symbol_remaining_quote_notional: Decimal
    portfolio_available_quote: Decimal
    portfolio_remaining_quote_notional: Decimal
    max_quote_notional: Decimal
    suggested_quote_notional: Decimal
    candidate_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        _require_decimal_fields(
            self,
            (
                "max_position_fraction",
                "symbol_current_quote_notional",
                "symbol_max_quote_notional",
                "symbol_remaining_quote_notional",
                "portfolio_available_quote",
                "portfolio_remaining_quote_notional",
                "max_quote_notional",
                "suggested_quote_notional",
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "max_position_fraction": str(self.max_position_fraction),
            "symbol_current_quote_notional": str(self.symbol_current_quote_notional),
            "symbol_max_quote_notional": str(self.symbol_max_quote_notional),
            "symbol_remaining_quote_notional": str(self.symbol_remaining_quote_notional),
            "portfolio_available_quote": str(self.portfolio_available_quote),
            "portfolio_remaining_quote_notional": str(self.portfolio_remaining_quote_notional),
            "max_quote_notional": str(self.max_quote_notional),
            "suggested_quote_notional": str(self.suggested_quote_notional),
            "candidate_count": self.candidate_count,
        }


@dataclass(frozen=True, slots=True)
class UnderwaterPositionSnapshot:
    """Per-position underwater and recovery eligibility state."""

    symbol: str
    base_quantity: Decimal
    quote_notional: Decimal
    cost_basis: Decimal | None
    reference_price: Decimal
    unrealized_pnl_fraction: Decimal | None
    underwater: bool
    recovery_eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        _require_decimal_fields(
            self,
            (
                "base_quantity",
                "quote_notional",
                "cost_basis",
                "reference_price",
                "unrealized_pnl_fraction",
            ),
            allow_none=True,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "base_quantity": str(self.base_quantity),
            "quote_notional": str(self.quote_notional),
            "cost_basis": _decimal_string_or_none(self.cost_basis),
            "reference_price": str(self.reference_price),
            "unrealized_pnl_fraction": _decimal_string_or_none(self.unrealized_pnl_fraction),
            "underwater": self.underwater,
            "recovery_eligible": self.recovery_eligible,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True, slots=True)
class UnderwaterStateSnapshot:
    """Portfolio underwater summary used before recovery intent phases."""

    reference_price: Decimal
    regime: MarketRegime
    positions: tuple[UnderwaterPositionSnapshot, ...]
    underwater_count: int
    recovery_eligible_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_decimal_fields(self, ("reference_price",))

    def to_payload(self) -> dict[str, object]:
        return {
            "reference_price": str(self.reference_price),
            "regime": self.regime.value,
            "underwater_count": self.underwater_count,
            "recovery_eligible_count": self.recovery_eligible_count,
            "reason_codes": self.reason_codes,
            "positions": [position.to_payload() for position in self.positions],
        }


@dataclass(frozen=True, slots=True)
class TargetIntent:
    """Execution-neutral desired strategy action for one symbol and timeframe."""

    intent_type: TargetIntentType
    execution_intent: TargetExecutionIntent
    symbol: str
    timeframe: str
    regime: MarketRegime
    side: GridLevelSide | None
    target_price: Decimal | None
    reference_price: Decimal
    price_band: PriceBand
    risk: StrategyRiskLimits
    guards: StrategyGuardSnapshot
    reason_codes: tuple[str, ...]
    grid_level_index: int | None = None
    position: PositionContext | None = None
    metadata: JsonPayload = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        if self.execution_intent not in _EXECUTION_INTENTS_BY_TYPE[self.intent_type]:
            raise ValueError("execution_intent is incompatible with intent_type")
        if not self.reason_codes:
            raise ValueError("reason_codes must not be empty")
        if self.intent_type is TargetIntentType.HOLD and self.side is not None:
            raise ValueError("hold intent side must be empty")
        if self.intent_type is TargetIntentType.ALERT and self.side is not None:
            raise ValueError("alert intent side must be empty")
        _require_decimal_fields(self, ("reference_price",))
        _require_decimal_fields(self, ("target_price",), allow_none=True)
        _reject_execution_detail_keys(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(_json_safe_mapping(self.metadata)))

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "intent_type": self.intent_type.value,
            "execution_intent": self.execution_intent.value,
            "strategy": "spot_grid",
            "regime": self.regime.value,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "reference_price": str(self.reference_price),
            "target_price": _decimal_string_or_none(self.target_price),
            "grid_level_index": self.grid_level_index,
            "side": self.side.value if self.side is not None else None,
            "price_band": self.price_band.to_payload(),
            "risk": self.risk.to_payload(),
            "guards": self.guards.to_payload(),
            "reason_codes": self.reason_codes,
        }
        if self.position is not None:
            payload["position"] = self.position.to_payload()
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """Pure Spot Grid strategy decision before platform signal persistence."""

    symbol: str
    timeframe: str
    regime: MarketRegime
    intents: tuple[TargetIntent, ...]
    diagnostics: JsonPayload = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        object.__setattr__(self, "diagnostics", MappingProxyType(_json_safe_mapping(self.diagnostics)))

    def to_payload(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "regime": self.regime.value,
            "intents": [intent.to_payload() for intent in self.intents],
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class SpotGridPlan:
    """Pure Spot Grid planning result."""

    symbol: str
    timeframe: str
    reference_price: Decimal
    range_low: Decimal
    range_high: Decimal
    levels: tuple[GridLevel, ...]
    diagnostics: dict[str, object]
    regime: MarketRegime = MarketRegime.RANGE
    intents: tuple[TargetIntent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", normalize_spot_grid_symbol(self.symbol))
        _require_decimal_fields(self, ("reference_price", "range_low", "range_high"))


def _decimal_string_or_none(value: Decimal | None) -> str | None:
    return _decimal_string(value) if value is not None else None


def _require_decimal_fields(instance: object, names: tuple[str, ...], *, allow_none: bool = False) -> None:
    for name in names:
        value = getattr(instance, name)
        if value is None and allow_none:
            continue
        if not isinstance(value, Decimal):
            raise TypeError(f"{type(instance).__name__}.{name} must be Decimal")


def _json_safe_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _json_safe_value(item) for key, item in value.items()}


def _reject_execution_detail_keys(value: Mapping[str, object]) -> None:
    for key, item in value.items():
        normalized_key = str(key).lower()
        if normalized_key in _EXECUTION_DETAIL_KEYS:
            raise ValueError("TargetIntent metadata must not contain execution detail fields")
        if isinstance(item, Mapping):
            _reject_execution_detail_keys(item)
        elif isinstance(item, tuple | list):
            for nested in item:
                if isinstance(nested, Mapping):
                    _reject_execution_detail_keys(nested)


def _json_safe_value(value: object) -> object:
    if isinstance(value, Decimal):
        return _decimal_string(value)
    if isinstance(value, float):
        raise TypeError("float values are not allowed in Spot Grid JSON payloads")
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, tuple | list):
        return tuple(_json_safe_value(item) for item in value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, bool | int | str) or value is None:
        return value
    raise TypeError(f"Unsupported Spot Grid JSON payload value type: {type(value).__name__}")


def _decimal_string(value: Decimal) -> str:
    return format(value, "f")
