from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from bot_platform_service.application import RuntimeCapabilities, RuntimeContextFactory, RuntimeContextRequest
from bot_platform_service.domain import (
    BotPermission,
    BotSignal,
    BotSignalPublishResult,
    BotSignalSide,
    BotSignalType,
    PermissionDeniedError,
    UnsupportedTimeframeError,
    normalize_timeframe,
    normalize_timeframes,
)


class _MarketData:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def get_latest_complete_snapshot(self, *, source, canonical_symbol, timeframe, min_snapshot_version=None):
        self.calls.append((canonical_symbol, timeframe))
        return object()

    async def build_context(self, *, source, canonical_symbol, primary_timeframe, supporting_timeframes):
        self.calls.append((canonical_symbol, primary_timeframe))
        return object()


class _SignalPublisher:
    def __init__(self) -> None:
        self.signals: list[BotSignal] = []

    async def publish(self, signal: BotSignal) -> BotSignalPublishResult:
        self.signals.append(signal)
        return BotSignalPublishResult(accepted=True, signal_id="signal-1")


class _StateStore:
    def __init__(self) -> None:
        self.saved: list[object] = []

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        return {"instance_id": instance_id, "namespace": namespace, "state_key": state_key}

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        self.saved.append((instance_id, namespace, state_key, value))


class _NotificationPublisher:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str, object]] = []

    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        self.notifications.append((instance_id, message_type, payload))


class _SecretProvider:
    async def resolve(self, *, instance_id: str, secret_name: str) -> str:
        return f"{instance_id}:{secret_name}"


class _Logger:
    def info(self, event: str, **fields: object) -> None:
        pass

    def warning(self, event: str, **fields: object) -> None:
        pass

    def error(self, event: str, **fields: object) -> None:
        pass


class _Metrics:
    def increment(self, name: str, *, tags: dict[str, str] | None = None) -> None:
        pass

    def observe(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        pass


class _Clock:
    def now(self) -> object:
        return datetime(2026, 7, 14, tzinfo=UTC)


def test_signal_and_notification_publishers_use_separate_permissions() -> None:
    signal_publisher = _SignalPublisher()
    notification_publisher = _NotificationPublisher()
    context = _context(
        permissions=frozenset({BotPermission.PUBLISH_SIGNALS}),
        signal_publisher=signal_publisher,
        notification_publisher=notification_publisher,
    )

    result = asyncio.run(context.signal_publisher.publish(_signal()))

    assert result.accepted is True
    assert len(signal_publisher.signals) == 1
    with pytest.raises(PermissionDeniedError) as exc_info:
        asyncio.run(context.notification_publisher.publish(instance_id="instance-1", message_type="alert", payload={}))
    assert exc_info.value.permission is BotPermission.SEND_NOTIFICATIONS
    assert notification_publisher.notifications == []


def test_notification_permission_does_not_allow_signal_publish() -> None:
    context = _context(permissions=frozenset({BotPermission.SEND_NOTIFICATIONS}))

    asyncio.run(context.notification_publisher.publish(instance_id="instance-1", message_type="alert", payload={}))
    with pytest.raises(PermissionDeniedError) as exc_info:
        asyncio.run(context.signal_publisher.publish(_signal()))

    assert exc_info.value.permission is BotPermission.PUBLISH_SIGNALS


def test_market_data_permission_controls_snapshot_provider() -> None:
    market_data = _MarketData()
    context = _context(permissions=frozenset({BotPermission.READ_MARKET_DATA}), market_data=market_data)

    asyncio.run(
        context.market_data.get_latest_complete_snapshot(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe="1h",
        )
    )

    assert market_data.calls == [("ETHUSDT", "1h")]


def test_missing_market_data_permission_raises_typed_error() -> None:
    context = _context(permissions=frozenset())

    with pytest.raises(PermissionDeniedError) as exc_info:
        asyncio.run(
            context.market_data.build_context(
                source="binance_spot",
                canonical_symbol="ETHUSDT",
                primary_timeframe="1h",
                supporting_timeframes=("4h",),
            )
        )

    assert exc_info.value.permission is BotPermission.READ_MARKET_DATA


def test_state_store_read_and_write_permissions_are_independent() -> None:
    state_store = _StateStore()
    read_context = _context(permissions=frozenset({BotPermission.READ_STATE}), state_store=state_store)
    write_context = _context(permissions=frozenset({BotPermission.WRITE_STATE}), state_store=state_store)

    loaded = asyncio.run(read_context.state_store.load(instance_id="instance-1", namespace="runtime", state_key="ETHUSDT"))
    assert loaded == {"instance_id": "instance-1", "namespace": "runtime", "state_key": "ETHUSDT"}

    with pytest.raises(PermissionDeniedError):
        asyncio.run(read_context.state_store.save(instance_id="instance-1", namespace="runtime", state_key="ETHUSDT", value={}))

    asyncio.run(write_context.state_store.save(instance_id="instance-1", namespace="runtime", state_key="ETHUSDT", value={"ok": True}))
    assert state_store.saved == [("instance-1", "runtime", "ETHUSDT", {"ok": True})]


def test_timeframe_aliases_are_normalized_centrally() -> None:
    assert normalize_timeframe("H1") == "1h"
    assert normalize_timeframe("4H") == "4h"
    assert normalize_timeframe("D1") == "1d"
    assert normalize_timeframes(("h1", "H4", "1D")) == ("1h", "4h", "1d")
    with pytest.raises(UnsupportedTimeframeError):
        normalize_timeframe("15m")


def _context(
    *,
    permissions: frozenset[BotPermission],
    market_data: _MarketData | None = None,
    signal_publisher: _SignalPublisher | None = None,
    state_store: _StateStore | None = None,
    notification_publisher: _NotificationPublisher | None = None,
):
    capabilities = RuntimeCapabilities(
        market_data=market_data or _MarketData(),
        signal_publisher=signal_publisher or _SignalPublisher(),
        state_store=state_store or _StateStore(),
        notification_publisher=notification_publisher or _NotificationPublisher(),
        secret_provider=_SecretProvider(),
        logger=_Logger(),
        metrics=_Metrics(),
        clock=_Clock(),
    )
    return RuntimeContextFactory().build(
        RuntimeContextRequest(
            instance_id="instance-1",
            permissions=permissions,
            capabilities=capabilities,
        )
    )


def _signal() -> BotSignal:
    return BotSignal.build(
        instance_id="instance-1",
        module_id="spot_grid_bot",
        symbol="ETHUSDT",
        timeframe="1h",
        snapshot_id="snapshot-1",
        signal_type=BotSignalType.ENTRY,
        side=BotSignalSide.BUY,
        confidence=Decimal("0.8"),
        reason="range_entry",
        payload_schema="spot_grid.range_entry",
        payload_schema_version=1,
        payload={"price": Decimal("100")},
    )
