from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from bot_platform_service.domain import (
    BotCandle,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunResult,
    BotRunStatus,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


def test_phase_35_zero_persistence_preview_returns_expected_intents_for_subscribed_symbols() -> None:
    preview = _ZeroPersistenceSpotGridPreview(
        adapter=SpotGridAdapter(
            cycle_service=SpotGridTradingCycleService(indicator_runtime=_PreviewIndicatorRuntime())
        )
    )

    results = asyncio.run(
        preview.run(
            instance_id="spot-grid-preview",
            symbols=("ETHUSDT", "BTCUSDT"),
            timeframe="1h",
        )
    )

    assert set(results) == {"ETHUSDT", "BTCUSDT"}
    assert all(result.status is BotRunStatus.COMPLETE for result in results.values())
    assert all(result.mode is BotMode.SIGNAL_ONLY for result in results.values())
    assert all(result.signals for result in results.values())
    assert {
        symbol: {signal.signal_type for signal in result.signals}
        for symbol, result in results.items()
    } == {
        "ETHUSDT": {BotSignalType.ENTRY},
        "BTCUSDT": {BotSignalType.ENTRY},
    }
    assert {
        symbol: {signal.symbol for signal in result.signals}
        for symbol, result in results.items()
    } == {
        "ETHUSDT": {"ETHUSDT"},
        "BTCUSDT": {"BTCUSDT"},
    }
    assert all(result.state_changes for result in results.values())
    assert all(result.diagnostics["subscribed_symbols"] == ("ETHUSDT", "BTCUSDT") for result in results.values())
    assert all(result.diagnostics["signal_count"] == len(result.signals) for result in results.values())
    assert preview.persistence.created_runs == []
    assert preview.persistence.persisted_signals == []
    assert preview.persistence.saved_state == []


def test_phase_35_zero_persistence_preview_does_not_call_orchestration_or_persistence_boundaries() -> None:
    preview = _ZeroPersistenceSpotGridPreview(
        adapter=SpotGridAdapter(
            cycle_service=SpotGridTradingCycleService(indicator_runtime=_PreviewIndicatorRuntime())
        )
    )

    asyncio.run(preview.run(instance_id="spot-grid-preview", symbols=("ETHUSDT",), timeframe="1h"))

    assert preview.persistence.called == []


@dataclass(slots=True)
class _ZeroPersistenceSpotGridPreview:
    adapter: SpotGridAdapter
    persistence: "_PersistenceSentinel" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.persistence is None:
            self.persistence = _PersistenceSentinel()

    async def run(
        self,
        *,
        instance_id: str,
        symbols: tuple[str, ...],
        timeframe: str,
    ) -> dict[str, BotRunResult]:
        results: dict[str, BotRunResult] = {}
        for symbol in symbols:
            request = BotRunRequest(
                run_id=f"preview-{symbol.lower()}",
                instance_id=instance_id,
                module_id="spot_grid",
                mode=BotMode.SIGNAL_ONLY,
                trigger_type=BotTriggerType.MANUAL,
                config={
                    "symbols": list(symbols),
                    "primary_timeframe": timeframe,
                    "supporting_timeframes": [],
                    "max_grid_levels": 2,
                    "max_position_fraction": "0.10",
                },
                market_data=BotMarketDataContext(primary_snapshot=_snapshot(symbol=symbol, timeframe=timeframe)),
            )
            results[symbol] = await self.adapter.run_once(request)
        return results


class _PersistenceSentinel:
    def __init__(self) -> None:
        self.called: list[str] = []
        self.created_runs: list[object] = []
        self.persisted_signals: list[object] = []
        self.saved_state: list[object] = []

    async def create_run(self, **kwargs) -> bool:
        self.called.append("create_run")
        self.created_runs.append(dict(kwargs))
        raise AssertionError("zero-persistence preview must not create run records")

    async def publish_signal(self, **kwargs) -> str:
        self.called.append("publish_signal")
        self.persisted_signals.append(dict(kwargs))
        raise AssertionError("zero-persistence preview must not persist signals")

    async def save(self, **kwargs) -> None:
        self.called.append("save")
        self.saved_state.append(dict(kwargs))
        raise AssertionError("zero-persistence preview must not save state")


class _PreviewIndicatorRuntime(FakeStockIndicatorsRuntime):
    def rsi_last(self, quotes, length: int):
        del quotes, length
        return Decimal("30")

    def realized_volatility_last(self, quotes, length: int):
        del quotes, length
        return Decimal("0.01")


def _snapshot(*, symbol: str, timeframe: str) -> BotMarketSnapshot:
    now = datetime(2026, 7, 26, tzinfo=UTC)
    candles = tuple(
        _candle(symbol=symbol, timeframe=timeframe, index=index, now=now)
        for index in range(30)
    )
    return BotMarketSnapshot(
        snapshot_id=f"preview-{symbol.lower()}-{timeframe}",
        source="binance_spot",
        canonical_symbol=symbol,
        provider_symbol=symbol,
        timeframe=timeframe,
        last_closed_candle_time=now + timedelta(hours=29),
        lookback_start_time=now,
        lookback_end_time=now + timedelta(hours=29),
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash=f"hash-preview-{symbol.lower()}-{timeframe}",
        candles=candles,
    )


def _candle(*, symbol: str, timeframe: str, index: int, now: datetime) -> BotCandle:
    opened_at = now + timedelta(hours=index)
    close = Decimal("100")
    return BotCandle(
        source="binance_spot",
        canonical_symbol=symbol,
        timeframe=timeframe,
        open_time=opened_at,
        close_time=opened_at + timedelta(hours=1),
        open=close,
        high=close + Decimal("5"),
        low=close - Decimal("5"),
        close=close,
        volume=Decimal("1000"),
    )
