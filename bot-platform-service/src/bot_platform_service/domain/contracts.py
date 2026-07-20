from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from bot_platform_service.domain.models import (
    BotHealth,
    BotInstanceConfig,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotRunRequest,
    BotRunResult,
    BotSignal,
    BotSignalPublishResult,
    BotStartRequest,
    BotStartResult,
    BotStopResult,
    BotValidationResult,
)


class MarketDataSnapshotProvider(Protocol):
    """Port for reading complete immutable market-data snapshots."""

    async def get_snapshot(self, *, snapshot_id: str) -> BotMarketSnapshot:
        """Return one immutable snapshot by id."""

    async def get_latest_complete_snapshot(
        self,
        *,
        source: str,
        canonical_symbol: str,
        timeframe: str,
        min_snapshot_version: int | None = None,
    ) -> BotMarketSnapshot:
        """Return the latest complete snapshot matching the request."""

    async def build_context(
        self,
        *,
        source: str,
        canonical_symbol: str,
        primary_timeframe: str,
        supporting_timeframes: tuple[str, ...],
    ) -> BotMarketDataContext:
        """Return primary and supporting snapshots for one bot run."""


class SignalPublisher(Protocol):
    """Port for idempotently publishing normalized bot signals."""

    async def publish(self, signal: BotSignal) -> BotSignalPublishResult:
        """Persist or accept one signal and return the publish outcome."""


class StateStore(Protocol):
    """Port for per-instance runtime state."""

    async def load(self, *, instance_id: str, namespace: str, state_key: str) -> object | None:
        """Load one state value for an instance."""

    async def save(self, *, instance_id: str, namespace: str, state_key: str, value: object) -> None:
        """Save one state value for an instance."""


class NotificationPublisher(Protocol):
    """Port for outbound bot notifications."""

    async def publish(self, *, instance_id: str, message_type: str, payload: object) -> None:
        """Publish one notification payload."""


class SecretProvider(Protocol):
    """Port for resolving instance-scoped secret references."""

    async def resolve(self, *, instance_id: str, secret_name: str) -> str:
        """Resolve one secret value for an instance."""


class StructuredLogger(Protocol):
    """Port for structured logging without binding to a concrete logger."""

    def info(self, event: str, **fields: object) -> None:
        """Record an informational event."""

    def warning(self, event: str, **fields: object) -> None:
        """Record a warning event."""

    def error(self, event: str, **fields: object) -> None:
        """Record an error event."""


class MetricsRecorder(Protocol):
    """Port for recording stable metrics."""

    def increment(self, name: str, *, tags: dict[str, str] | None = None) -> None:
        """Increment one counter metric."""

    def observe(self, name: str, value: float, *, tags: dict[str, str] | None = None) -> None:
        """Observe one numeric metric value."""


class Clock(Protocol):
    """Port for time access."""

    def now(self) -> object:
        """Return the current time."""


@dataclass(frozen=True, slots=True)
class BotRuntimeContext:
    """Runtime capabilities passed to a bot module."""

    market_data: MarketDataSnapshotProvider
    signal_publisher: SignalPublisher
    state_store: StateStore
    notification_publisher: NotificationPublisher
    secret_provider: SecretProvider
    logger: StructuredLogger
    metrics: MetricsRecorder
    clock: Clock


class BotModule(Protocol):
    """Protocol implemented by concrete bot module adapters."""

    module_id: str

    async def validate_config(self, config: BotInstanceConfig) -> BotValidationResult:
        """Validate one bot instance configuration."""

    async def initialize(self, context: BotRuntimeContext) -> None:
        """Initialize the bot module with platform-provided capabilities."""

    async def dry_run(self, request: BotRunRequest) -> BotRunResult:
        """Run one non-persisting planning pass."""

    async def run_once(self, request: BotRunRequest) -> BotRunResult:
        """Run one platform-controlled signal generation pass."""

    async def start(self, request: BotStartRequest) -> BotStartResult:
        """Start a long-running bot instance when supported."""

    async def stop(self, instance_id: str) -> BotStopResult:
        """Stop a long-running bot instance when supported."""

    async def health(self, instance_id: str) -> BotHealth:
        """Return current bot instance health."""
