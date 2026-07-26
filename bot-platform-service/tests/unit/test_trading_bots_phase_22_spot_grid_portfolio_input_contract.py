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
    BotSignalType,
    BotTriggerType,
)
from bot_platform_service.trading_bots.spot_grid.adapter import SpotGridAdapter
from bot_platform_service.trading_bots.spot_grid.application import SpotGridTradingCycleService
from bot_platform_service.trading_bots.spot_grid.domain import PortfolioContext, PositionContext
from tests.fixtures.spot_grid_indicator_runtime import FakeStockIndicatorsRuntime


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
CONTRACT_DOC = SERVICE_ROOT / "docs" / "spot_grid_portfolio_position_input_contract.md"


def test_phase_22_missing_portfolio_context_uses_conservative_empty_fallback() -> None:
    adapter = _adapter()

    result = asyncio.run(adapter.run_once(_request(config={"supporting_timeframes": [], "max_grid_levels": 2})))

    assert result.status is BotRunStatus.COMPLETE
    assert result.signals == ()
    assert result.diagnostics["portfolio_context"] == {
        "provided": False,
        "fallback": "empty_conservative",
        "total_equity": "0",
        "available_quote": "0",
        "position_count": 0,
        "position_symbols": (),
    }
    assert result.diagnostics["no_loss"]["block_reason"] == "position_context_missing"


def test_phase_22_portfolio_context_provider_supplies_position_input_to_strategy() -> None:
    provider = _PortfolioContextProvider(
        PortfolioContext(
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
    )
    adapter = _adapter(portfolio_context_provider=provider)

    result = asyncio.run(adapter.run_once(_request(config={"supporting_timeframes": [], "max_grid_levels": 2})))

    assert result.status is BotRunStatus.COMPLETE
    assert provider.calls == [("instance-1", "ETHUSDT", "1h", "snapshot-phase-22")]
    assert {signal.signal_type for signal in result.signals} == {BotSignalType.EXIT}
    assert result.diagnostics["portfolio_context"]["provided"] is True
    assert result.diagnostics["portfolio_context"]["position_symbols"] == ("ETHUSDT",)
    assert result.diagnostics["no_loss"]["cost_basis"] == "100"


def test_phase_22_portfolio_position_input_contract_is_documented() -> None:
    doc = CONTRACT_DOC.read_text(encoding="utf-8")

    assert "`PortfolioContext` and `PositionContext` as explicit platform\n  inputs" in doc
    assert "`SpotGridPortfolioContextProvider.get_portfolio_context(...)`" in doc
    assert '"positions": []' in doc
    assert "empty_conservative" in doc
    assert "must not query private exchanges" in doc


def test_phase_22_spot_grid_has_no_direct_exchange_or_account_query_terms() -> None:
    forbidden_terms = (
        "ccxt",
        "pybit",
        "Bybit",
        "Binance",
        "fetch_balance",
        "get_wallet_balance",
        "get_positions",
        "private_key",
        "api_secret",
        "place_order",
        "create_order",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GRID_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


class _PortfolioContextProvider:
    def __init__(self, portfolio_context: PortfolioContext | None) -> None:
        self.portfolio_context = portfolio_context
        self.calls: list[tuple[str, str, str, str]] = []

    async def get_portfolio_context(
        self,
        *,
        instance_id: str,
        symbol: str,
        timeframe: str,
        snapshot_id: str,
    ) -> PortfolioContext | None:
        self.calls.append((instance_id, symbol, timeframe, snapshot_id))
        return self.portfolio_context


class _ExitRuntime(FakeStockIndicatorsRuntime):
    def rsi_last(self, quotes, length: int):
        del quotes, length
        return Decimal("70")

    def realized_volatility_last(self, quotes, length: int):
        del quotes, length
        return Decimal("0.01")


def _adapter(portfolio_context_provider: _PortfolioContextProvider | None = None) -> SpotGridAdapter:
    return SpotGridAdapter(
        cycle_service=SpotGridTradingCycleService(indicator_runtime=_ExitRuntime()),
        portfolio_context_provider=portfolio_context_provider,
    )


def _request(*, config: dict[str, object]) -> BotRunRequest:
    return BotRunRequest(
        run_id="run-phase-22",
        instance_id="instance-1",
        module_id="spot_grid",
        mode=BotMode.SIGNAL_ONLY,
        trigger_type=BotTriggerType.MANUAL,
        config=config,
        market_data=BotMarketDataContext(primary_snapshot=_snapshot()),
    )


def _snapshot() -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    candle = BotCandle(
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
    )
    return BotMarketSnapshot(
        snapshot_id="snapshot-phase-22",
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe="1h",
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash="hash-phase-22",
        candles=(candle,),
    )
