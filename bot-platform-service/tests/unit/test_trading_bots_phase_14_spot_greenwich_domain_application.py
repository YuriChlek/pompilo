from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from bot_platform_service.domain import BotCandle, BotMarketDataContext, BotMarketSnapshot
from bot_platform_service.trading_bots.spot_greenwich.application import (
    GreenwichTradingCycleService,
    parse_greenwich_config,
)
from bot_platform_service.trading_bots.spot_greenwich.domain import (
    GreenwichActionType,
    GreenwichCandle,
    GreenwichConfig,
    GreenwichExecutionConfig,
    GreenwichPositionState,
    GreenwichSignalConfig,
    GreenwichSignalType,
    GreenwichSpotPlanner,
    GreenwichSpotSignal,
    apply_portfolio_position_limit,
    decide_spot_execution,
    generate_spot_signal,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GREENWICH_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_greenwich"


def test_phase_14_spot_greenwich_domain_generates_decimal_buy_and_sell_signals() -> None:
    config = _signal_config()

    buy_signal = generate_spot_signal("ethusdt", _buy_candles(), config=config)
    sell_signal = generate_spot_signal("ETHUSDT", _sell_candles(), config=config)

    assert buy_signal.symbol == "ETHUSDT"
    assert buy_signal.signal_type is GreenwichSignalType.BUY
    assert buy_signal.reason == "greenwich_buy_recovery"
    assert isinstance(buy_signal.signal_price, Decimal)
    assert sell_signal.signal_type is GreenwichSignalType.SELL
    assert sell_signal.reason == "greenwich_sell_fade"


def test_phase_14_spot_greenwich_execution_policy_matches_legacy_sizing_rules() -> None:
    signal = GreenwichSpotSignal("ETHUSDT", GreenwichSignalType.BUY, Decimal("100"), "2026-07-15", "test")
    first_entry_state = GreenwichPositionState("ETHUSDT", Decimal("0"), Decimal("0"), Decimal("0"))
    second_entry_state = GreenwichPositionState("ETHUSDT", Decimal("1"), Decimal("110"), Decimal("110"), entry_count=1)

    first = decide_spot_execution(signal, first_entry_state, Decimal("500"))
    second = decide_spot_execution(
        GreenwichSpotSignal("ETHUSDT", GreenwichSignalType.BUY, Decimal("90"), "2026-07-15", "test"),
        second_entry_state,
        Decimal("500"),
    )

    assert first.action is GreenwichActionType.BUY
    assert first.quote_amount == Decimal("25")
    assert first.quantity == Decimal("0.25000000")
    assert second.action is GreenwichActionType.BUY
    assert second.quote_amount == Decimal("15")
    assert second.quantity == Decimal("0.16666666")


def test_phase_14_spot_greenwich_portfolio_limit_blocks_low_priority_new_entries() -> None:
    decisions = {
        "BTCUSDT": _decision("BTCUSDT"),
        "ETHUSDT": _decision("ETHUSDT"),
        "SUIUSDT": _decision("SUIUSDT"),
    }
    position_states = {
        "BTCUSDT": _empty_position("BTCUSDT"),
        "ETHUSDT": _empty_position("ETHUSDT"),
        "SUIUSDT": _empty_position("SUIUSDT"),
        "SOLUSDT": GreenwichPositionState("SOLUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
        "XRPUSDT": GreenwichPositionState("XRPUSDT", Decimal("1"), Decimal("100"), Decimal("100")),
    }

    constrained = apply_portfolio_position_limit(
        decisions,
        position_states,
        config=GreenwichExecutionConfig(portfolio_position_limit=3, portfolio_priority_symbols=("BTCUSDT", "ETHUSDT")),
    )

    assert constrained["BTCUSDT"].action is GreenwichActionType.BUY
    assert constrained["ETHUSDT"].action is GreenwichActionType.SKIP
    assert constrained["ETHUSDT"].reason == "portfolio_position_limit_priority_blocked"
    assert constrained["SUIUSDT"].action is GreenwichActionType.SKIP


def test_phase_14_spot_greenwich_planner_builds_one_timeframe_plan() -> None:
    config = GreenwichConfig(
        symbols=("ETHUSDT",),
        primary_timeframe="1d",
        supporting_timeframes=("4h",),
        signal=_signal_config(confirmation_candle_enabled=False),
    )
    planner = GreenwichSpotPlanner(config)

    plan = planner.plan(
        symbol="ETHUSDT",
        candles=_buy_candles(),
        position_state=_empty_position("ETHUSDT"),
        available_quote_balance=Decimal("500"),
        timeframe="1d",
    )

    assert plan.signal.signal_type is GreenwichSignalType.BUY
    assert plan.decision.action is GreenwichActionType.BUY
    assert plan.diagnostics["planner"] == "spot_greenwich_decimal_bands"
    assert plan.diagnostics["candle_count"] == 5


def test_phase_14_spot_greenwich_application_runs_one_cycle_from_platform_snapshot() -> None:
    config = parse_greenwich_config(
        {
            "symbols": ["ETHUSDT"],
            "primary_timeframe": "1d",
            "supporting_timeframes": ["4h"],
            "greenwich_length": 3,
            "greenwich_multiplier_1": "0.5",
            "greenwich_multiplier_2": "0.5",
            "greenwich_multiplier_3": "0.5",
            "confirmation_candle_enabled": False,
            "atr_position_sizing_enabled": False,
        },
        fallback_symbols=("BTCUSDT",),
        fallback_timeframes=("1d", "4h"),
    )
    service = GreenwichTradingCycleService()

    result = service.run_once(
        market_data=BotMarketDataContext(primary_snapshot=_snapshot("snapshot-1", "1d", _buy_candles())),
        config=config,
    )

    assert result.plan.signal.signal_type is GreenwichSignalType.BUY
    assert result.plan.decision.action is GreenwichActionType.BUY
    assert result.diagnostics["snapshot_id"] == "snapshot-1"
    assert result.diagnostics["data_hash"] == "hash-snapshot-1"


def test_phase_14_spot_greenwich_config_parser_uses_platform_fallbacks() -> None:
    config = parse_greenwich_config({}, fallback_symbols=("ETHUSDT",), fallback_timeframes=("1d", "4h"))

    assert config.symbols == ("ETHUSDT",)
    assert config.primary_timeframe == "1d"
    assert config.supporting_timeframes == ("4h",)
    assert config.signal.length == 98
    assert config.execution.deposit_percent == Decimal("5")


def test_phase_14_spot_greenwich_source_boundary_has_no_private_exchange_or_sync_imports() -> None:
    forbidden_terms = (
        "spot-greenwich-bot",
        "spot_greenwich_bot",
        "asyncpg",
        "sqlalchemy",
        "Bybit",
        "Binance",
        "ensure_candle_tables",
        "MarketDataSynchronizer",
        "scheduler",
        "command_dispatch",
        "run_migrations",
        "place_order",
        "cancel_order",
        "create_market_buy_order",
        "create_market_sell_order",
    )
    violations: list[str] = []
    for path in sorted(SPOT_GREENWICH_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        found = [term for term in forbidden_terms if term in text]
        if found:
            violations.append(f"{path.relative_to(SERVICE_ROOT)}: {', '.join(found)}")

    assert violations == []


def _signal_config(*, confirmation_candle_enabled: bool = False) -> GreenwichSignalConfig:
    return GreenwichSignalConfig(
        length=3,
        multiplier_1=Decimal("0.5"),
        multiplier_2=Decimal("0.5"),
        multiplier_3=Decimal("0.5"),
        confirmation_candle_enabled=confirmation_candle_enabled,
        atr_position_sizing_enabled=False,
    )


def _buy_candles() -> tuple[GreenwichCandle, ...]:
    return (
        _candle(1, "100"),
        _candle(2, "100"),
        _candle(3, "100"),
        _candle(4, "100", low="90"),
        _candle(5, "105", low="103"),
    )


def _sell_candles() -> tuple[GreenwichCandle, ...]:
    return (
        _candle(1, "100"),
        _candle(2, "100"),
        _candle(3, "100"),
        _candle(4, "120"),
        _candle(5, "100"),
    )


def _candle(index: int, close: str, *, low: str | None = None, high: str | None = None) -> GreenwichCandle:
    close_price = Decimal(close)
    low_price = Decimal(low) if low is not None else close_price - Decimal("1")
    high_price = Decimal(high) if high is not None else close_price + Decimal("1")
    return GreenwichCandle(
        timestamp=f"2026-07-{10 + index:02d}T00:00:00+00:00",
        open=close_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=Decimal("1000"),
    )


def _empty_position(symbol: str) -> GreenwichPositionState:
    return GreenwichPositionState(symbol, Decimal("0"), Decimal("0"), Decimal("0"))


def _decision(symbol: str):
    signal = GreenwichSpotSignal(symbol, GreenwichSignalType.BUY, Decimal("100"), "2026-07-15", "test")
    return decide_spot_execution(signal, _empty_position(symbol), Decimal("500"))


def _snapshot(snapshot_id: str, timeframe: str, candles: tuple[GreenwichCandle, ...]) -> BotMarketSnapshot:
    now = datetime(2026, 7, 15, tzinfo=UTC)
    platform_candles = tuple(
        BotCandle(
            source="binance_spot",
            canonical_symbol="ETHUSDT",
            timeframe=timeframe,
            open_time=now + timedelta(minutes=index),
            close_time=now + timedelta(minutes=index + 1),
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
            volume=candle.volume,
        )
        for index, candle in enumerate(candles)
    )
    return BotMarketSnapshot(
        snapshot_id=snapshot_id,
        source="binance_spot",
        canonical_symbol="ETHUSDT",
        provider_symbol="ETHUSDT",
        timeframe=timeframe,
        last_closed_candle_time=now,
        lookback_start_time=now,
        lookback_end_time=now,
        completeness_status="COMPLETE",
        snapshot_version=1,
        data_hash=f"hash-{snapshot_id}",
        candles=platform_candles,
    )
