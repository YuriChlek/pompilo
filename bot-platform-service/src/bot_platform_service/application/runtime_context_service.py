from __future__ import annotations

from dataclasses import dataclass

from bot_platform_service.domain import (
    BotPermission,
    BotRuntimeContext,
    MarketDataSnapshotProvider,
    MetricsRecorder,
    NotificationPublisher,
    PermissionDeniedError,
    SecretProvider,
    SignalPublisher,
    StateStore,
    StructuredLogger,
    Clock,
)
from bot_platform_service.domain.models import BotMarketDataContext, BotMarketSnapshot, BotSignal, BotSignalPublishResult


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Concrete capabilities available for one runtime context."""

    market_data: MarketDataSnapshotProvider
    signal_publisher: SignalPublisher
    state_store: StateStore
    notification_publisher: NotificationPublisher
    secret_provider: SecretProvider
    logger: StructuredLogger
    metrics: MetricsRecorder
    clock: Clock


@dataclass(frozen=True, slots=True)
class RuntimeContextRequest:
    """Input for building an instance-scoped runtime context."""

    instance_id: str
    permissions: frozenset[BotPermission]
    capabilities: RuntimeCapabilities


class RuntimeContextFactory:
    """Build permission-scoped runtime contexts for bot modules."""

    def build(self, request: RuntimeContextRequest) -> BotRuntimeContext:
        """Return a runtime context whose controlled capabilities enforce permissions."""
        permissions = request.permissions
        instance_id = request.instance_id
        capabilities = request.capabilities
        return BotRuntimeContext(
            market_data=_PermissionedMarketDataSnapshotProvider(
                capabilities.market_data,
                instance_id=instance_id,
                permissions=permissions,
            ),
            signal_publisher=_PermissionedSignalPublisher(
                capabilities.signal_publisher,
                instance_id=instance_id,
                permissions=permissions,
            ),
            state_store=_PermissionedStateStore(
                capabilities.state_store,
                instance_id=instance_id,
                permissions=permissions,
            ),
            notification_publisher=_PermissionedNotificationPublisher(
                capabilities.notification_publisher,
                instance_id=instance_id,
                permissions=permissions,
            ),
            secret_provider=capabilities.secret_provider,
            logger=capabilities.logger,
            metrics=capabilities.metrics,
            clock=capabilities.clock,
        )


def _require_permission(instance_id: str, permissions: frozenset[BotPermission], permission: BotPermission) -> None:
    if permission not in permissions:
        raise PermissionDeniedError(permission, instance_id=instance_id)


@dataclass(frozen=True, slots=True)
class _PermissionedMarketDataSnapshotProvider:
    delegate: MarketDataSnapshotProvider
    instance_id: str
    permissions: frozenset[BotPermission]

    async def get_latest_complete_snapshot(
        self,
        *,
        source: str,
        canonical_symbol: str,
        timeframe: str,
        min_snapshot_version: int | None = None,
    ) -> BotMarketSnapshot:
        _require_permission(self.instance_id, self.permissions, BotPermission.READ_MARKET_DATA)
        return await self.delegate.get_latest_complete_snapshot(
            source=source,
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            min_snapshot_version=min_snapshot_version,
        )

    async def build_context(
        self,
        *,
        source: str,
        canonical_symbol: str,
        primary_timeframe: str,
        supporting_timeframes: tuple[str, ...],
    ) -> BotMarketDataContext:
        _require_permission(self.instance_id, self.permissions, BotPermission.READ_MARKET_DATA)
        return await self.delegate.build_context(
            source=source,
            canonical_symbol=canonical_symbol,
            primary_timeframe=primary_timeframe,
            supporting_timeframes=supporting_timeframes,
        )


@dataclass(frozen=True, slots=True)
class _PermissionedSignalPublisher:
    delegate: SignalPublisher
    instance_id: str
    permissions: frozenset[BotPermission]

    async def publish(self, signal: BotSignal) -> BotSignalPublishResult:
        _require_permission(self.instance_id, self.permissions, BotPermission.PUBLISH_SIGNALS)
        return await self.delegate.publish(signal)


@dataclass(frozen=True, slots=True)
class _PermissionedStateStore:
    delegate: StateStore
    instance_id: str
    permissions: frozenset[BotPermission]

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        _require_permission(self.instance_id, self.permissions, BotPermission.READ_STATE)
        return await self.delegate.load(instance_id=instance_id, namespace=namespace, state_key=state_key)

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        _require_permission(self.instance_id, self.permissions, BotPermission.WRITE_STATE)
        await self.delegate.save(instance_id=instance_id, namespace=namespace, state_key=state_key, value=value)


@dataclass(frozen=True, slots=True)
class _PermissionedNotificationPublisher:
    delegate: NotificationPublisher
    instance_id: str
    permissions: frozenset[BotPermission]

    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        _require_permission(self.instance_id, self.permissions, BotPermission.SEND_NOTIFICATIONS)
        await self.delegate.publish(instance_id=instance_id, message_type=message_type, payload=payload)
