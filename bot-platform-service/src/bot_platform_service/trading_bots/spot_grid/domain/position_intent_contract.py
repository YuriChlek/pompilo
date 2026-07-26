from __future__ import annotations

POSITION_INTENT_PAYLOAD_SCHEMA = "spot_grid.position_intent"
POSITION_INTENT_PAYLOAD_SCHEMA_VERSION = 1

POSITION_INTENT_ENTRY_EXAMPLE: dict[str, object] = {
    "intent_type": "open_position",
    "execution_intent": "limit_entry_candidate",
    "strategy": "spot_grid",
    "regime": "range",
    "symbol": "ETHUSDT",
    "timeframe": "1h",
    "reference_price": "3202.10",
    "target_price": "3180.50",
    "grid_level_index": 3,
    "side": "buy",
    "price_band": {
        "range_low": "3100.00",
        "range_high": "3350.00",
    },
    "risk": {
        "max_position_fraction": "0.10",
        "suggested_quote_notional": "125.00",
        "max_quote_notional": "150.00",
    },
    "guards": {
        "rsi14": "31.5",
        "atr14": "42.0",
        "buy_allowed": True,
        "sell_allowed": False,
        "no_loss_required": True,
        "no_loss_passed": None,
    },
    "reason_codes": ("range_buy", "rsi_buy_passed"),
}

POSITION_INTENT_EXIT_EXAMPLE: dict[str, object] = {
    "intent_type": "close_position",
    "execution_intent": "limit_exit_candidate",
    "strategy": "spot_grid",
    "regime": "range",
    "symbol": "ETHUSDT",
    "timeframe": "1h",
    "reference_price": "3288.40",
    "target_price": "3295.00",
    "grid_level_index": 4,
    "side": "sell",
    "price_band": {
        "range_low": "3100.00",
        "range_high": "3350.00",
    },
    "position": {
        "cost_basis": "3265.75",
        "min_no_loss_exit_price": "3265.75",
    },
    "guards": {
        "rsi14": "68.0",
        "sell_allowed": True,
        "no_loss_passed": True,
    },
    "reason_codes": ("range_take_profit", "rsi_sell_passed", "no_loss_passed"),
}

POSITION_INTENT_REBALANCE_EXAMPLE: dict[str, object] = {
    "intent_type": "rebalance",
    "execution_intent": "rebalance_candidate",
    "strategy": "spot_grid",
    "regime": "high_volatility",
    "symbol": "ETHUSDT",
    "timeframe": "1h",
    "reference_price": "3250.00",
    "target_price": "3244.00",
    "grid_level_index": None,
    "side": "sell",
    "price_band": {
        "range_low": "3180.00",
        "range_high": "3330.00",
    },
    "risk": {
        "max_position_fraction": "0.06",
        "suggested_quote_notional": None,
        "max_quote_notional": "90.00",
    },
    "guards": {
        "rsi14": "54.0",
        "atr14": "78.0",
        "buy_allowed": False,
        "sell_allowed": True,
        "no_loss_required": True,
        "no_loss_passed": True,
    },
    "reason_codes": ("volatility_rebalance", "position_exposure_reduce"),
}

POSITION_INTENT_HOLD_EXAMPLE: dict[str, object] = {
    "intent_type": "hold",
    "execution_intent": "no_action",
    "strategy": "spot_grid",
    "regime": "downtrend",
    "symbol": "ETHUSDT",
    "timeframe": "1h",
    "reference_price": "3190.00",
    "target_price": None,
    "grid_level_index": None,
    "side": None,
    "price_band": {
        "range_low": "3150.00",
        "range_high": "3280.00",
    },
    "risk": {
        "max_position_fraction": "0.10",
        "suggested_quote_notional": None,
        "max_quote_notional": None,
    },
    "guards": {
        "rsi14": "44.0",
        "atr14": "62.0",
        "buy_allowed": False,
        "sell_allowed": False,
        "no_loss_required": False,
        "no_loss_passed": None,
    },
    "reason_codes": ("supporting_4h_downtrend_block", "hold_without_new_entry"),
}

POSITION_INTENT_ALERT_EXAMPLE: dict[str, object] = {
    "intent_type": "alert",
    "execution_intent": "operator_alert",
    "strategy": "spot_grid",
    "regime": "risk_off",
    "symbol": "ETHUSDT",
    "timeframe": "1h",
    "reference_price": "3120.00",
    "target_price": None,
    "grid_level_index": None,
    "side": None,
    "price_band": {
        "range_low": "3090.00",
        "range_high": "3275.00",
    },
    "risk": {
        "max_position_fraction": "0.00",
        "suggested_quote_notional": None,
        "max_quote_notional": None,
    },
    "guards": {
        "rsi14": "22.0",
        "atr14": "140.0",
        "buy_allowed": False,
        "sell_allowed": False,
        "no_loss_required": False,
        "no_loss_passed": None,
    },
    "reason_codes": ("risk_off_alert", "manual_review_required"),
}

POSITION_INTENT_EXAMPLES: tuple[dict[str, object], ...] = (
    POSITION_INTENT_ENTRY_EXAMPLE,
    POSITION_INTENT_EXIT_EXAMPLE,
    POSITION_INTENT_REBALANCE_EXAMPLE,
    POSITION_INTENT_HOLD_EXAMPLE,
    POSITION_INTENT_ALERT_EXAMPLE,
)


__all__ = [
    "POSITION_INTENT_ENTRY_EXAMPLE",
    "POSITION_INTENT_EXIT_EXAMPLE",
    "POSITION_INTENT_REBALANCE_EXAMPLE",
    "POSITION_INTENT_HOLD_EXAMPLE",
    "POSITION_INTENT_ALERT_EXAMPLE",
    "POSITION_INTENT_EXAMPLES",
    "POSITION_INTENT_PAYLOAD_SCHEMA",
    "POSITION_INTENT_PAYLOAD_SCHEMA_VERSION",
]
