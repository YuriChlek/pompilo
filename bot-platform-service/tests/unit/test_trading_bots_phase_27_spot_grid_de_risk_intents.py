from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from bot_platform_service.trading_bots.spot_grid.domain import (
    GridLevelSide,
    IndicatorSnapshot,
    MarketRegime,
    PositionContext,
    SpotGridCandle,
    SpotGridConfig,
    SpotGridPlanner,
    TargetExecutionIntent,
    TargetIntentType,
    target_intent_to_bot_signal,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
DE_RISK_SOURCE = (
    SERVICE_ROOT
    / "src"
    / "bot_platform_service"
    / "trading_bots"
    / "spot_grid"
    / "domain"
    / "de_risk.py"
)


def test_phase_27_de_risk_sell_can_bypass_rsi_sell_rule_in_risk_off() -> None:
    plan = _risk_off_plan(rsi14=Decimal("30"), position_context=_position_context())

    de_risk_intent = _single_de_risk_intent(plan.intents)
    payload = de_risk_intent.to_payload()

    assert de_risk_intent.intent_type is TargetIntentType.CLOSE_POSITION
    assert de_risk_intent.execution_intent is TargetExecutionIntent.DE_RISK_CANDIDATE
    assert de_risk_intent.side is GridLevelSide.SELL
    assert de_risk_intent.target_price == Decimal("100.00000000")
    assert payload["guards"]["sell_allowed"] is True
    assert payload["guards"]["no_loss_required"] is False
    assert payload["metadata"]["rsi_sell_rule_bypassed"] is True
    assert "rsi_sell_rule_bypassed" in payload["reason_codes"]
    assert plan.diagnostics["rsi_sell_allowed"] is False
    assert plan.diagnostics["de_risk"]["intent_count"] == 1


def test_phase_27_de_risk_is_separate_from_ordinary_take_profit_exit() -> None:
    risk_off = _risk_off_plan(rsi14=Decimal("70"), position_context=_position_context())
    range_take_profit = SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "112", "90", "100"),),
        config=_config(),
        regime=MarketRegime.RANGE,
        indicators=_indicators(rsi14=Decimal("70")),
        position_context=_position_context(),
    )

    de_risk_intent = _single_de_risk_intent(risk_off.intents)
    ordinary_exit = next(intent for intent in range_take_profit.intents if "range_take_profit" in intent.reason_codes)

    assert de_risk_intent.execution_intent is TargetExecutionIntent.DE_RISK_CANDIDATE
    assert "de_risk_sell" in de_risk_intent.reason_codes
    assert "range_take_profit" not in de_risk_intent.reason_codes
    assert ordinary_exit.execution_intent is TargetExecutionIntent.LIMIT_EXIT_CANDIDATE
    assert "range_take_profit" in ordinary_exit.reason_codes
    assert "de_risk_sell" not in ordinary_exit.reason_codes


def test_phase_27_de_risk_does_not_emit_without_open_position() -> None:
    no_position = _risk_off_plan(rsi14=Decimal("30"), position_context=None)
    zero_position = _risk_off_plan(
        rsi14=Decimal("30"),
        position_context=PositionContext(
            symbol="ETHUSDT",
            base_quantity=Decimal("0"),
            quote_notional=Decimal("0"),
            cost_basis=Decimal("100"),
        ),
    )

    assert _de_risk_intents(no_position.intents) == ()
    assert no_position.diagnostics["de_risk"]["reason_codes"] == (
        "de_risk_position_missing",
        "de_risk_not_eligible",
    )
    assert _de_risk_intents(zero_position.intents) == ()
    assert zero_position.diagnostics["de_risk"]["has_open_position"] is False


def test_phase_27_de_risk_signal_payload_keeps_execution_safety_recheck_marker() -> None:
    plan = _risk_off_plan(rsi14=Decimal("30"), position_context=_position_context())
    signal = target_intent_to_bot_signal(
        _single_de_risk_intent(plan.intents),
        instance_id="instance-1",
        module_id="spot_grid",
        snapshot_id="snapshot-phase-27",
        confidence=Decimal("0.60"),
    )

    assert signal.payload["execution_intent"] == "de_risk_candidate"
    assert signal.payload["metadata"]["execution_risk_recheck_required"] is True
    assert signal.reason == "de_risk_sell"
    assert signal.payload["reason_codes"] == (
        "de_risk_sell",
        "risk_off_de_risk",
        "position_exposure_detected",
        "execution_risk_recheck_required",
        "rsi_sell_rule_bypassed",
    )


def test_phase_27_de_risk_intent_payload_has_no_private_execution_details() -> None:
    payload = _single_de_risk_intent(_risk_off_plan(rsi14=Decimal("30"), position_context=_position_context()).intents).to_payload()
    private_terms = (
        "client_order_id",
        "order_id",
        "exchange_order_id",
        "venue_order_id",
        "fill_id",
        "fill_state",
        "filled_size",
        "api_key",
        "api_secret",
        "private_key",
    )

    assert _private_payload_terms(payload, private_terms) == []


def test_phase_27_de_risk_source_has_no_exchange_or_persistence_reads() -> None:
    text = DE_RISK_SOURCE.read_text(encoding="utf-8")
    forbidden_terms = (
        "asyncpg",
        "sqlalchemy",
        "requests",
        "httpx",
        "ccxt",
        "pybit",
        "fetch_balance",
        "get_wallet_balance",
        "get_positions",
        "place_order",
        "create_order",
    )

    assert [term for term in forbidden_terms if term in text] == []


def _risk_off_plan(*, rsi14: Decimal, position_context: PositionContext | None):
    return SpotGridPlanner().plan(
        symbol="ETHUSDT",
        timeframe="1h",
        candles=(_candle("100", "110", "90", "100"),),
        config=_config(),
        regime=MarketRegime.RISK_OFF,
        indicators=_indicators(rsi14=rsi14),
        position_context=position_context,
    )


def _de_risk_intents(intents):
    return tuple(intent for intent in intents if intent.execution_intent is TargetExecutionIntent.DE_RISK_CANDIDATE)


def _single_de_risk_intent(intents):
    de_risk_intents = _de_risk_intents(intents)
    assert len(de_risk_intents) == 1
    return de_risk_intents[0]


def _position_context() -> PositionContext:
    return PositionContext(
        symbol="ETHUSDT",
        base_quantity=Decimal("1"),
        quote_notional=Decimal("100"),
        cost_basis=Decimal("100"),
    )


def _config() -> SpotGridConfig:
    return SpotGridConfig(
        symbols=("ETHUSDT",),
        primary_timeframe="1h",
        supporting_timeframes=(),
        max_position_fraction=Decimal("0.10"),
        max_grid_levels=2,
    )


def _indicators(*, rsi14: Decimal) -> IndicatorSnapshot:
    return IndicatorSnapshot(
        ema20=Decimal("100"),
        ema50=Decimal("100"),
        ema200=Decimal("100"),
        atr14=Decimal("2"),
        rsi14=rsi14,
        realized_volatility=Decimal("0.01"),
        realized_volatility_short=Decimal("0.01"),
        current_volume=Decimal("1000"),
        volume_ma20=Decimal("1000"),
        volume_ratio=Decimal("1"),
        candle_count=30,
        has_required_history=True,
        volatility_has_required_history=True,
        volume_has_required_history=True,
    )


def _candle(open_price: str, high: str, low: str, close: str) -> SpotGridCandle:
    return SpotGridCandle(
        timestamp="2026-07-15T00:00:00+00:00",
        open=Decimal(open_price),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def _private_payload_terms(value, private_terms: tuple[str, ...]) -> list[str]:
    if isinstance(value, dict):
        matches = [key for key in value if key in private_terms]
        for item in value.values():
            matches.extend(_private_payload_terms(item, private_terms))
        return matches
    if isinstance(value, list | tuple):
        matches: list[str] = []
        for item in value:
            matches.extend(_private_payload_terms(item, private_terms))
        return matches
    return []
