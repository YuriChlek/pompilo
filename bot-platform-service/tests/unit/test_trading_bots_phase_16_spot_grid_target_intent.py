from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from bot_platform_service.domain import BotSignalSide, BotSignalType, canonical_json
from bot_platform_service.trading_bots.spot_grid.domain import (
    POSITION_INTENT_EXAMPLES,
    POSITION_INTENT_PAYLOAD_SCHEMA,
    POSITION_INTENT_PAYLOAD_SCHEMA_VERSION,
    GridLevelSide,
    MarketRegime,
    PriceBand,
    StrategyGuardSnapshot,
    StrategyRiskLimits,
    TargetExecutionIntent,
    TargetIntent,
    TargetIntentType,
    target_intent_to_bot_signal,
)


SERVICE_ROOT = Path(__file__).resolve().parents[2]
SPOT_GRID_ROOT = SERVICE_ROOT / "src" / "bot_platform_service" / "trading_bots" / "spot_grid"
CONTRACT_DOC = SERVICE_ROOT / "docs" / "spot_grid_position_intent_contract.md"


def test_phase_16_target_intent_contract_documents_all_intent_types() -> None:
    examples_by_type = {str(example["intent_type"]): example for example in POSITION_INTENT_EXAMPLES}

    assert set(examples_by_type) == {"open_position", "close_position", "rebalance", "hold", "alert"}
    for intent_type, example in examples_by_type.items():
        assert example["execution_intent"]
        assert "target_price" in example
        assert example["reason_codes"]
        assert canonical_json(example)
        assert _forbidden_execution_terms(str(example)) == []
        if intent_type in {"hold", "alert"}:
            assert example["side"] is None
            assert example["target_price"] is None


def test_phase_16_target_intent_rejects_execution_specific_metadata() -> None:
    for key in ("order_id", "exchange_order_id", "venue_order_id", "fill_id", "fill_state"):
        with pytest.raises(ValueError, match="execution detail fields"):
            TargetIntent(**_intent_kwargs(metadata={key: "private-execution-state"}))

    with pytest.raises(ValueError, match="execution detail fields"):
        TargetIntent(**_intent_kwargs(metadata={"nested": [{"filled_size": "1.5"}]}))


def test_phase_16_target_intent_requires_reason_codes_and_compatible_execution_intent() -> None:
    with pytest.raises(ValueError, match="reason_codes must not be empty"):
        TargetIntent(**_intent_kwargs(reason_codes=()))

    with pytest.raises(ValueError, match="execution_intent is incompatible"):
        TargetIntent(
            **_intent_kwargs(
                intent_type=TargetIntentType.HOLD,
                execution_intent=TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE,
                side=None,
                target_price=None,
            )
        )

    with pytest.raises(ValueError, match="hold intent side must be empty"):
        TargetIntent(
            **_intent_kwargs(
                intent_type=TargetIntentType.HOLD,
                execution_intent=TargetExecutionIntent.NO_ACTION,
                side=GridLevelSide.BUY,
                target_price=None,
                reason_codes=("hold",),
            )
        )


def test_phase_16_target_intent_converts_to_bot_signal_without_execution_api() -> None:
    cases = (
        (
            TargetIntent(**_intent_kwargs()),
            BotSignalType.ENTRY,
            BotSignalSide.BUY,
            "range_buy",
        ),
        (
            TargetIntent(
                **_intent_kwargs(
                    intent_type=TargetIntentType.CLOSE_POSITION,
                    execution_intent=TargetExecutionIntent.LIMIT_EXIT_CANDIDATE,
                    side=GridLevelSide.SELL,
                    reason_codes=("range_take_profit",),
                )
            ),
            BotSignalType.EXIT,
            BotSignalSide.SELL,
            "range_take_profit",
        ),
        (
            TargetIntent(
                **_intent_kwargs(
                    intent_type=TargetIntentType.REBALANCE,
                    execution_intent=TargetExecutionIntent.REBALANCE_CANDIDATE,
                    side=GridLevelSide.SELL,
                    reason_codes=("volatility_rebalance",),
                )
            ),
            BotSignalType.REBALANCE,
            BotSignalSide.SELL,
            "volatility_rebalance",
        ),
        (
            TargetIntent(
                **_intent_kwargs(
                    intent_type=TargetIntentType.HOLD,
                    execution_intent=TargetExecutionIntent.NO_ACTION,
                    side=None,
                    target_price=None,
                    reason_codes=("hold_without_new_entry",),
                )
            ),
            BotSignalType.HOLD,
            None,
            "hold_without_new_entry",
        ),
        (
            TargetIntent(
                **_intent_kwargs(
                    intent_type=TargetIntentType.ALERT,
                    execution_intent=TargetExecutionIntent.OPERATOR_ALERT,
                    side=None,
                    target_price=None,
                    reason_codes=("risk_off_alert",),
                )
            ),
            BotSignalType.ALERT,
            None,
            "risk_off_alert",
        ),
    )

    for intent, expected_signal_type, expected_side, expected_reason in cases:
        signal = target_intent_to_bot_signal(
            intent,
            instance_id="instance-1",
            module_id="spot_grid",
            snapshot_id="snapshot-1",
            confidence=Decimal("0.60"),
        )

        assert signal.payload_schema == POSITION_INTENT_PAYLOAD_SCHEMA
        assert signal.payload_schema_version == POSITION_INTENT_PAYLOAD_SCHEMA_VERSION
        assert signal.signal_type is expected_signal_type
        assert signal.side is expected_side
        assert signal.reason == expected_reason
        assert signal.payload["intent_type"] == intent.intent_type.value
        assert _forbidden_execution_terms(str(signal.payload)) == []


def test_phase_16_target_intent_documentation_and_source_replace_target_order_concept() -> None:
    doc = CONTRACT_DOC.read_text(encoding="utf-8")
    assert "`intent_type` is one of `open_position`, `close_position`, `rebalance`, `hold`, or" in doc
    assert "`execution_intent` is an execution-neutral hint" in doc
    assert "Payloads do not include venue execution identifiers" in doc
    assert "accepted, rejected, placed,\n  filled, cancelled, or failed" in doc

    source_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(SPOT_GRID_ROOT.rglob("*.py")))
    assert "TargetOrder" not in source_text
    assert "target_order_builder" not in source_text


def _intent_kwargs(
    *,
    intent_type: TargetIntentType = TargetIntentType.OPEN_POSITION,
    execution_intent: TargetExecutionIntent = TargetExecutionIntent.LIMIT_ENTRY_CANDIDATE,
    side: GridLevelSide | None = GridLevelSide.BUY,
    target_price: Decimal | None = Decimal("100.25"),
    reason_codes: tuple[str, ...] = ("range_buy",),
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "intent_type": intent_type,
        "execution_intent": execution_intent,
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "regime": MarketRegime.RANGE,
        "side": side,
        "target_price": target_price,
        "reference_price": Decimal("102.50"),
        "price_band": PriceBand(range_low=Decimal("95"), range_high=Decimal("110")),
        "risk": StrategyRiskLimits(max_position_fraction=Decimal("0.10")),
        "guards": StrategyGuardSnapshot(buy_allowed=side is GridLevelSide.BUY, sell_allowed=side is GridLevelSide.SELL),
        "reason_codes": reason_codes,
        "grid_level_index": 1 if side is not None else None,
        "metadata": metadata or {},
    }


def _forbidden_execution_terms(value: str) -> list[str]:
    forbidden_terms = (
        "client_order_id",
        "order_id",
        "exchange_order_id",
        "venue_order_id",
        "fill_id",
        "fill_state",
        "filled_size",
    )
    return [term for term in forbidden_terms if term in value]
