from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from application.trading_cycle_service import SpotTradingCycleService
from domain.inventory_models import InventorySnapshot
from domain.market_models import Candle, MarketContext, VenueConstraints
from domain.order_models import TargetOrder
from domain.portfolio_allocator import PortfolioAllocator
from domain.spot_grid_planner import SpotGridPlanner
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG

if TYPE_CHECKING:
    from bot_platform_service.domain.contracts import MarketDataSnapshotProvider, SignalPublisher
    from bot_platform_service.domain.models import BotMarketSnapshot


@dataclass(slots=True)
class PlatformSpotGridDependencies:
    """Injected platform capabilities for signal-only grid bot composition."""

    market_data_snapshot_provider: MarketDataSnapshotProvider
    signal_publisher: SignalPublisher


@dataclass(slots=True)
class PlatformSpotGridComposition:
    """Composition result used by the Bot Platform adapter layer."""

    trading_cycle: SpotTradingCycleService
    market_data_provider: PlatformSpotGridMarketDataProvider
    executor: SignalOnlyOrderExecutor
    notifier: PlatformSignalNotifier


class PlatformSpotGridMarketDataProvider:
    """Adapt platform immutable snapshots to the grid bot market-data port."""

    def __init__(
        self,
        snapshot_provider: MarketDataSnapshotProvider,
        *,
        source: str = "binance_spot",
        timeframe: str = "1h",
        higher_timeframe: str = "4h",
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.source = source
        self.timeframe = timeframe
        self.higher_timeframe = higher_timeframe

    async def get_market_context(self, symbol: str, *, persisted_cost_basis: float | None = None) -> MarketContext:
        """Load platform snapshots and expose a legacy-compatible planning context."""

        primary = await self.snapshot_provider.get_latest_complete_snapshot(
            source=self.source,
            canonical_symbol=symbol,
            timeframe=self.timeframe,
        )
        higher = await self.snapshot_provider.get_latest_complete_snapshot(
            source=self.source,
            canonical_symbol=symbol,
            timeframe=self.higher_timeframe,
        )
        mark_price = float(primary.candles[-1].close) if primary.candles else 0.0
        return MarketContext(
            symbol=symbol.upper(),
            candles=_to_grid_candles(primary),
            inventory=InventorySnapshot(
                base_balance=0.0,
                quote_balance=0.0,
                reserved_quote=0.0,
                mark_price=mark_price,
                cost_basis_price=persisted_cost_basis,
            ),
            live_orders=[],
            venue_constraints=VenueConstraints(
                tick_size=0.0,
                qty_step=0.0,
                min_order_qty=0.0,
                min_order_amt=0.0,
            ),
            higher_timeframe_candles=_to_grid_candles(higher),
        )


@dataclass(slots=True)
class SignalOnlyOrderExecutor:
    """No-exchange execution bridge that records target-order decisions only."""

    signal_publisher: SignalPublisher
    target_orders_by_symbol: dict[str, tuple[TargetOrder, ...]] = field(default_factory=dict)

    async def reconcile_state(self, symbols) -> None:
        """Platform mode does not reconcile private exchange state."""

    async def sync_orders(self, symbol: str, target_orders: list[TargetOrder]) -> bool:
        """Capture decisions for signal mapping without creating exchange orders."""

        self.target_orders_by_symbol[symbol.upper()] = tuple(target_orders)
        return True


@dataclass(slots=True)
class PlatformSignalNotifier:
    """Notification seam that keeps access to the injected platform signal publisher."""

    signal_publisher: SignalPublisher
    decisions: list[object] = field(default_factory=list)

    async def notify_rebuild(self, decision) -> None:
        """Record rebuild decisions for the adapter-level signal mapper."""

        self.decisions.append(decision)


def build_platform_trading_cycle(
    dependencies: PlatformSpotGridDependencies,
    *,
    strategy_config=DEFAULT_STRATEGY_CONFIG,
) -> PlatformSpotGridComposition:
    """Compose the grid bot for Bot Platform mode without DB or exchange clients."""

    market_data_provider = PlatformSpotGridMarketDataProvider(dependencies.market_data_snapshot_provider)
    executor = SignalOnlyOrderExecutor(dependencies.signal_publisher)
    notifier = PlatformSignalNotifier(dependencies.signal_publisher)
    trading_cycle = SpotTradingCycleService(
        market_data_provider=market_data_provider,
        executor=executor,
        notifier=notifier,
        planner=SpotGridPlanner(strategy_config),
        state_store=None,
        portfolio_allocator=PortfolioAllocator(strategy_config),
    )
    return PlatformSpotGridComposition(
        trading_cycle=trading_cycle,
        market_data_provider=market_data_provider,
        executor=executor,
        notifier=notifier,
    )


def _to_grid_candles(snapshot: BotMarketSnapshot) -> list[Candle]:
    return [
        Candle(
            timestamp=_timestamp_millis(candle.close_time),
            open=float(candle.open),
            high=float(candle.high),
            low=float(candle.low),
            close=float(candle.close),
            volume=float(candle.volume),
        )
        for candle in snapshot.candles
    ]


def _timestamp_millis(value: datetime) -> int:
    return int(value.timestamp() * 1000)


__all__ = [
    "PlatformSignalNotifier",
    "PlatformSpotGridComposition",
    "PlatformSpotGridDependencies",
    "PlatformSpotGridMarketDataProvider",
    "SignalOnlyOrderExecutor",
    "build_platform_trading_cycle",
]
