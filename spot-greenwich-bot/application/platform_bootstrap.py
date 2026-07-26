from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

from application.trading_cycle_service import TradingCycleService
from domain.models import ExecutionDecision, ExecutionResult, PositionState, SpotSignal
from domain.planner import MultiTimeframeSpotPlanner

if TYPE_CHECKING:
    from bot_platform_service.domain.contracts import MarketDataSnapshotProvider, SignalPublisher
    from bot_platform_service.domain.models import BotMarketSnapshot


@dataclass(slots=True)
class PlatformGreenwichDependencies:
    """Injected platform capabilities for signal-only Greenwich composition."""

    market_data_snapshot_provider: MarketDataSnapshotProvider
    signal_publisher: SignalPublisher


@dataclass(slots=True)
class PlatformGreenwichComposition:
    """Composition result used by the Bot Platform adapter layer."""

    trading_cycle: TradingCycleService
    market_data_provider: PlatformGreenwichMarketDataProvider
    executor: SignalOnlyPositionExecutor
    notifier: PlatformSignalNotifier


class PlatformGreenwichMarketDataProvider:
    """Adapt platform immutable snapshots to the Greenwich market-data port."""

    def __init__(
        self,
        snapshot_provider: MarketDataSnapshotProvider,
        *,
        source: str = "binance_spot",
        d1_timeframe: str = "1d",
        h4_timeframe: str = "4h",
    ) -> None:
        self.snapshot_provider = snapshot_provider
        self.source = source
        self.d1_timeframe = d1_timeframe
        self.h4_timeframe = h4_timeframe

    async def get_symbol_history_async(self, symbol: str) -> dict[str, object]:
        """Load D1 and H4 snapshots from the platform snapshot provider."""

        d1_snapshot = await self.snapshot_provider.get_latest_complete_snapshot(
            source=self.source,
            canonical_symbol=symbol,
            timeframe=self.d1_timeframe,
        )
        h4_snapshot = await self.snapshot_provider.get_latest_complete_snapshot(
            source=self.source,
            canonical_symbol=symbol,
            timeframe=self.h4_timeframe,
        )
        return {
            "d1": _to_dataframe(d1_snapshot),
            "h4": _to_dataframe(h4_snapshot),
        }

    def get_symbol_history(self, symbol: str) -> dict[str, object]:
        """Require async loading in platform mode to avoid hidden event-loop work."""

        raise RuntimeError("Use get_symbol_history_async in platform mode")


@dataclass(slots=True)
class SignalOnlyPositionExecutor:
    """No-exchange position executor used by platform-mode composition."""

    signal_publisher: SignalPublisher
    default_quote_balance: Decimal = Decimal("0")
    position_states: dict[str, PositionState] = field(default_factory=dict)
    execution_results: list[ExecutionResult] = field(default_factory=list)

    async def get_position_state(self, symbol: str) -> PositionState:
        """Return injected state or a zero-position default without exchange reconciliation."""

        normalized_symbol = symbol.upper()
        return self.position_states.get(
            normalized_symbol,
            PositionState(
                symbol=normalized_symbol,
                quantity=Decimal("0"),
                avg_entry_price=Decimal("0"),
                total_cost=Decimal("0"),
                entry_count=0,
                first_take_profit_done=False,
            ),
        )

    async def get_quote_balance(self, symbol: str) -> Decimal:
        """Return platform-provided dry quote capacity without private exchange clients."""

        return self.default_quote_balance

    async def execute(
        self,
        decision: ExecutionDecision,
        position_state: PositionState,
        *,
        dry_run: bool = False,
    ) -> ExecutionResult:
        """Capture execution decisions for signal mapping without opening orders."""

        result = ExecutionResult(
            executed=False,
            symbol=decision.symbol,
            action=decision.action,
            reason=f"platform_signal_only:{decision.reason}",
            signal_price=decision.signal_price,
            executed_price=None,
            quantity=decision.quantity,
            exchange_order_id=None,
            dry_run=dry_run,
            notification_only=True,
        )
        self.execution_results.append(result)
        return result


@dataclass(slots=True)
class PlatformSignalNotifier:
    """Notification seam that keeps access to the injected platform signal publisher."""

    signal_publisher: SignalPublisher
    notifications: list[tuple[SpotSignal, ExecutionResult]] = field(default_factory=list)

    async def notify(self, signal: SpotSignal, result: ExecutionResult) -> None:
        """Record generated signal/result pairs for adapter-level signal mapping."""

        self.notifications.append((signal, result))


class PlatformGreenwichTradingCycleService(TradingCycleService):
    """Async snapshot-aware trading cycle for platform-mode Greenwich runs."""

    async def run(self, symbol: str, dry_run: bool = True) -> dict:
        candles = await self.market_data_provider.get_symbol_history_async(symbol)
        position_state = await self.executor.get_position_state(symbol.upper())
        available_quote_balance = await self.executor.get_quote_balance(symbol.upper())
        plan = self.planner.plan(symbol.upper(), candles, position_state, available_quote_balance)
        result = await self.execution_service.execute(
            plan.signal,
            plan.decision,
            position_state,
            dry_run=dry_run,
        )
        return {
            "signal": plan.signal,
            "decision": plan.decision,
            "result": result,
            "position_state": position_state,
        }


def build_platform_trading_cycle(
    dependencies: PlatformGreenwichDependencies,
    *,
    default_quote_balance: Decimal = Decimal("0"),
    d1_regime_filter_enabled: bool = True,
) -> PlatformGreenwichComposition:
    """Compose Greenwich for Bot Platform mode without DB or exchange clients."""

    market_data_provider = PlatformGreenwichMarketDataProvider(dependencies.market_data_snapshot_provider)
    executor = SignalOnlyPositionExecutor(
        dependencies.signal_publisher,
        default_quote_balance=default_quote_balance,
    )
    notifier = PlatformSignalNotifier(dependencies.signal_publisher)
    trading_cycle = PlatformGreenwichTradingCycleService(
        market_data_provider=market_data_provider,
        executor=executor,
        notifier=notifier,
        planner=MultiTimeframeSpotPlanner(d1_regime_filter_enabled=d1_regime_filter_enabled),
    )
    return PlatformGreenwichComposition(
        trading_cycle=trading_cycle,
        market_data_provider=market_data_provider,
        executor=executor,
        notifier=notifier,
    )


def _to_dataframe(snapshot: BotMarketSnapshot):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "open_time": candle.open_time,
                "close_time": candle.close_time,
                "symbol": snapshot.canonical_symbol,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            }
            for candle in snapshot.candles
        ]
    )


__all__ = [
    "PlatformGreenwichComposition",
    "PlatformGreenwichDependencies",
    "PlatformGreenwichMarketDataProvider",
    "PlatformGreenwichTradingCycleService",
    "PlatformSignalNotifier",
    "SignalOnlyPositionExecutor",
    "build_platform_trading_cycle",
]
