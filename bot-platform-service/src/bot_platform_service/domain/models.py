from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from bot_platform_service.domain.enums import (
    BotHealthStatus,
    BotMode,
    BotModuleStatus,
    BotNotificationStatus,
    BotRunStatus,
    BotSignalSide,
    BotSignalType,
    BotStateChangeOperation,
    BotTriggerType,
)

JsonMapping = Mapping[str, object]


def _normalize_for_json(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize_for_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple | list):
        return [_normalize_for_json(item) for item in value]
    if isinstance(value, bool | int | str) or value is None:
        return value
    raise TypeError(f"Unsupported payload value type: {type(value).__name__}")


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, tuple | list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _freeze_json_mapping(value: Mapping[str, object]) -> JsonMapping:
    return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})


def canonical_json(value: Mapping[str, object]) -> str:
    """Return stable JSON used for payload hashes and signal keys."""
    normalized = _normalize_for_json(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_payload_hash(payload: Mapping[str, object]) -> str:
    """Build a deterministic SHA-256 hash for a signal payload."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_signal_key(
    *,
    instance_id: str,
    module_id: str,
    symbol: str,
    timeframe: str,
    snapshot_id: str,
    signal_type: BotSignalType,
    side: BotSignalSide | None,
    payload_schema_version: int,
    payload_hash: str,
) -> str:
    """Build the deterministic idempotency key for one signal."""
    key_payload = {
        "instance_id": instance_id,
        "module_id": module_id,
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "snapshot_id": snapshot_id,
        "signal_type": signal_type.value,
        "side": side.value if side is not None else None,
        "payload_schema_version": payload_schema_version,
        "payload_hash": payload_hash,
    }
    return hashlib.sha256(canonical_json(key_payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class BotManifest:
    module_id: str
    display_name: str
    version: str
    supported_modes: tuple[BotMode, ...]
    required_timeframes: tuple[str, ...]
    required_market_data: tuple[str, ...]
    supports_multi_symbol: bool
    config_schema_version: int
    status: BotModuleStatus = BotModuleStatus.ACTIVE


@dataclass(frozen=True, slots=True)
class BotModuleMetadata:
    module_id: str
    display_name: str
    version: str
    adapter_path: str
    adapter_class: str | None
    status: BotModuleStatus
    manifest: JsonMapping
    config_schema_version: int | None = None
    config_schema: JsonMapping | None = None


@dataclass(frozen=True, slots=True)
class BotInstanceConfig:
    instance_id: str
    module_id: str
    mode: BotMode
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    config_schema_version: int
    config: JsonMapping = field(default_factory=dict)
    tenant_id: str | None = None
    name: str | None = None


@dataclass(frozen=True, slots=True)
class BotCandle:
    source: str
    canonical_symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True, slots=True)
class BotMarketSnapshot:
    snapshot_id: str
    source: str
    canonical_symbol: str
    provider_symbol: str
    timeframe: str
    last_closed_candle_time: datetime
    lookback_start_time: datetime
    lookback_end_time: datetime
    completeness_status: str
    snapshot_version: int
    data_hash: str
    candles: tuple[BotCandle, ...]


@dataclass(frozen=True, slots=True)
class BotMarketDataContext:
    primary_snapshot: BotMarketSnapshot
    supporting_snapshots: tuple[BotMarketSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class BotSignal:
    signal_key: str
    instance_id: str
    module_id: str
    symbol: str
    timeframe: str
    snapshot_id: str
    signal_type: BotSignalType
    side: BotSignalSide | None
    confidence: Decimal | None
    reason: str
    payload_schema: str
    payload_schema_version: int
    payload_hash: str
    payload: JsonMapping

    @classmethod
    def build(
        cls,
        *,
        instance_id: str,
        module_id: str,
        symbol: str,
        timeframe: str,
        snapshot_id: str,
        signal_type: BotSignalType,
        side: BotSignalSide | None,
        confidence: Decimal | None,
        reason: str,
        payload_schema: str,
        payload_schema_version: int,
        payload: Mapping[str, object],
    ) -> "BotSignal":
        payload_hash = build_payload_hash(payload)
        signal_key = build_signal_key(
            instance_id=instance_id,
            module_id=module_id,
            symbol=symbol,
            timeframe=timeframe,
            snapshot_id=snapshot_id,
            signal_type=signal_type,
            side=side,
            payload_schema_version=payload_schema_version,
            payload_hash=payload_hash,
        )
        return cls(
            signal_key=signal_key,
            instance_id=instance_id,
            module_id=module_id,
            symbol=symbol.upper(),
            timeframe=timeframe,
            snapshot_id=snapshot_id,
            signal_type=signal_type,
            side=side,
            confidence=confidence,
            reason=reason,
            payload_schema=payload_schema,
            payload_schema_version=payload_schema_version,
            payload_hash=payload_hash,
            payload=MappingProxyType(dict(payload)),
        )


@dataclass(frozen=True, slots=True)
class BotSignalPublishResult:
    accepted: bool
    signal_id: str | None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class BotNotificationEvent:
    notification_id: str
    instance_id: str
    module_id: str
    status: BotNotificationStatus
    channel: str
    message_type: str
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class BotStateChange:
    instance_id: str
    namespace: str
    state_key: str
    operation: BotStateChangeOperation
    value: JsonMapping | None


@dataclass(frozen=True, slots=True)
class BotRuntimeStateRecord:
    instance_id: str
    namespace: str
    state_key: str
    state_json: JsonMapping
    state_hash: str
    version: int
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class BotRunEventRecord:
    event_id: str
    run_id: str
    instance_id: str
    module_id: str
    event_type: str
    payload_json: JsonMapping
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class BotRecoverableRunRecord:
    run_id: str
    instance_id: str
    module_id: str
    status: BotRunStatus
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class BotRunRequest:
    run_id: str
    instance_id: str
    module_id: str
    mode: BotMode
    trigger_type: BotTriggerType
    config: JsonMapping = field(default_factory=dict)
    market_data: BotMarketDataContext | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", _freeze_json_mapping(self.config))


@dataclass(frozen=True, slots=True)
class BotStartRequest:
    instance_id: str
    module_id: str
    mode: BotMode
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class BotStartResult:
    accepted: bool
    instance_id: str
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class BotStopResult:
    accepted: bool
    instance_id: str
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class BotValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BotHealth:
    instance_id: str
    module_id: str
    status: BotHealthStatus
    details: JsonMapping = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BotRunResult:
    run_id: str
    instance_id: str
    module_id: str
    mode: BotMode
    status: BotRunStatus
    signals: tuple[BotSignal, ...] = ()
    notifications: tuple[BotNotificationEvent, ...] = ()
    diagnostics: JsonMapping = field(default_factory=dict)
    state_changes: tuple[BotStateChange, ...] = ()
    error_code: str | None = None
    error_message_redacted: str | None = None
