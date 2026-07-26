from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import (
    BotCandle,
    BotMarketDataContext,
    BotMarketSnapshot,
    BotMode,
    BotRunRequest,
    BotRunStatus,
    BotSignalSide,
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import (
    POSITION_INTENT_PAYLOAD_SCHEMA,
    POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
    PortfolioContext,
    PositionContext,
)
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"


def test_phase_28_adapter_converts_target_intents_to_position_intent_signals() -> None:
    result = asyncio.run(_adapter().run_once(_request(run_id="run-phase-28-a")))

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals
    assert {signal.signal_type for signal in result.signals} == {BotSignalType.EXIT}
    assert {signal.side for signal in result.signals} == {BotSignalSide.SELL}

    signal = result.signals[0]
    assert signal.payload_schema == POSITION_INTENT_PAYLOAD_SCHEMA
    assert signal.payload_schema_version == POSITION_INTENT_PAYLOAD_SCHEMA_VERSION
    assert signal.reason == signal.payload["reason_codes"][0]
    assert signal.payload["strategy"] == "spot_grid"
    assert signal.payload["intent_type"] == "close_position"
    assert signal.payload["execution_intent"] == "limit_exit_candidate"
    assert signal.payload["price_band"]
    assert signal.payload["risk"]
    assert signal.payload["guards"]
    assert signal.payload["position"]
    assert signal.payload["reason_codes"]
    assert signal.payload_schema != "spot_grid.grid_level"


def test_phase_28_bot_signal_build_creates_deterministic_signal_key_from_intent_payload() -> None:
    first = asyncio.run(_adapter().run_once(_request(run_id="run-phase-28-a")))
    second = asyncio.run(_adapter().run_once(_request(run_id="run-phase-28-b")))

    assert [(signal.signal_key, signal.payload_hash) for signal in first.signals] == [
        (signal.signal_key, signal.payload_hash) for signal in second.signals
    ]


def test_phase_28_spot_grid_source_no_longer_uses_grid_level_signal_builder() -> None:
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(SPOT_GRID_ROOT.rglob("*.py")))

    assert "_signal_from_level" not in source_text
    assert "spot_grid.grid_level" not in source_text


class _PortfolioContextProvider:
    async def get_portfolio_context(
        self,
        *,
        instance_id: str,
        symbol: str,
        timeframe: str,
        snapshot_id: str,
    ) -> PortfolioContext:
        del instance_id, symbol, timeframe, snapshot_id
        return PortfolioContext(
            total_equity=Decimal("1000"),
            available_quote=Decimal("500"),
            positions=(
                PositionContext(
                    symbol="ETHUSDT",
                    base_quantity=Decimal("1"),
                    quote_notional=Decimal("100"),
                    cost_basis=Decimal("100"),
                ),
            ),
        )


class _ExitRuntime(FakeStockIndicatorsRuntime):
    def rsi_last(self, quotes, length: int):
        del quotes, length
        return Decimal("70")

    def realized_volatility_last(self, quotes, length: int):
        del quotes, length
        return Decimal("0.01")


def _adapter() -> SpotGridAdapter:
    return SpotGridAdapter(
        cycle_service=SpotGridTradingCycleService(indicator_runtime=_ExitRuntime()),
        portfolio_context_provider=_PortfolioContextProvider(),
    )


def _request(*, run_id: str) -> BotRunRequest:
    return BotRunRequest(
        run_id=run_id,
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.MANUAL,
        config={"supporting_timeframes": [], "max_grid_levels": 2},
        market_data=BotMarketDataContext(primary_snapshot=_snapshot()),
    )


def _snapshot() -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candles = (
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe="1h",
            open_time=now,
            close_time=now,
            open=Decimal("100"),
            high=Decimal("112"),
            low=Decimal("90"),
            close=Decimal("100"),
            volume=Decimal("1000"),
        ),
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-28",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-28",
        candles=candles,
    )
