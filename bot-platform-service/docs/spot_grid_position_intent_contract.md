# Spot Grid Position Intent Contract

This contract defines the execution-facing payload produced by the platform-native
`spot_grid` strategy when a planned grid level becomes a persisted `BotSignal`.

## Payload Identity

- `payload_schema`: `spot_grid.position_intent`
- `payload_schema_version`: `1`

The schema version is part of the `BotSignal.signal_key`. Any semantic payload shape
change must increment `payload_schema_version`.

## Payload Rules

- Payloads must be JSON-safe under `BotSignal` constraints.
- Decimal trading values are encoded as strings.
- Symbols use canonical uppercase format such as `ETHUSDT`.
- `intent_type` is one of `open_position`, `close_position`, `rebalance`, `hold`, or
  `alert`.
- `execution_intent` is an execution-neutral hint such as `limit_entry_candidate`,
  `limit_exit_candidate`, `rebalance_candidate`, `no_action`, or `operator_alert`.
- Every payload includes `target_price`; `hold` and `alert` intents set it to `null`.
- Every payload includes `reason_codes`; these are strategy diagnostics, not execution
  status events.
- Entry payloads describe an `open_position` target intent.
- Exit payloads describe a `close_position` target intent.
- Rebalance payloads describe a strategy-level exposure adjustment candidate.
- Hold payloads document a deliberate no-action decision.
- Alert payloads document a strategy condition that should be surfaced without execution.
- Payloads do not include venue execution identifiers, fills, account secrets, or private
  exchange credentials.
- Payloads must not include order lifecycle state such as accepted, rejected, placed,
  filled, cancelled, or failed. That state belongs to a separate execution service.

## Field Summary

- `intent_type`: strategy action category.
- `execution_intent`: execution-neutral hint for downstream services.
- `strategy`: fixed `spot_grid`.
- `regime`: effective Spot Grid market regime.
- `symbol`: canonical platform symbol.
- `timeframe`: primary timeframe used for the intent.
- `reference_price`: latest planning reference price.
- `target_price`: target price for actionable intents, or `null` for no-action/alert.
- `grid_level_index`: grid level when applicable, otherwise `null`.
- `side`: `buy`, `sell`, or `null`.
- `price_band`: range context used by the strategy.
- `risk`: execution-neutral risk hints, never venue sizing or private account data.
- `guards`: strategy guardrail values and booleans.
- `position`: optional execution-neutral position context for exits.
- `reason_codes`: stable strategy reasons for the target intent.

## Entry Example

```json
{
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
    "range_high": "3350.00"
  },
  "risk": {
    "max_position_fraction": "0.10",
    "suggested_quote_notional": "125.00",
    "max_quote_notional": "150.00"
  },
  "guards": {
    "rsi14": "31.5",
    "atr14": "42.0",
    "buy_allowed": true,
    "sell_allowed": false,
    "no_loss_required": true,
    "no_loss_passed": null
  },
  "reason_codes": ["range_buy", "rsi_buy_passed"]
}
```

## Exit Example

```json
{
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
    "range_high": "3350.00"
  },
  "position": {
    "cost_basis": "3265.75",
    "min_no_loss_exit_price": "3265.75"
  },
  "guards": {
    "rsi14": "68.0",
    "sell_allowed": true,
    "no_loss_passed": true
  },
  "reason_codes": ["range_take_profit", "rsi_sell_passed", "no_loss_passed"]
}
```

## Rebalance Example

```json
{
  "intent_type": "rebalance",
  "execution_intent": "rebalance_candidate",
  "strategy": "spot_grid",
  "regime": "high_volatility",
  "symbol": "ETHUSDT",
  "timeframe": "1h",
  "reference_price": "3250.00",
  "target_price": "3244.00",
  "grid_level_index": null,
  "side": "sell",
  "price_band": {
    "range_low": "3180.00",
    "range_high": "3330.00"
  },
  "risk": {
    "max_position_fraction": "0.06",
    "suggested_quote_notional": null,
    "max_quote_notional": "90.00"
  },
  "guards": {
    "rsi14": "54.0",
    "atr14": "78.0",
    "buy_allowed": false,
    "sell_allowed": true,
    "no_loss_required": true,
    "no_loss_passed": true
  },
  "reason_codes": ["volatility_rebalance", "position_exposure_reduce"]
}
```

## Hold Example

```json
{
  "intent_type": "hold",
  "execution_intent": "no_action",
  "strategy": "spot_grid",
  "regime": "downtrend",
  "symbol": "ETHUSDT",
  "timeframe": "1h",
  "reference_price": "3190.00",
  "target_price": null,
  "grid_level_index": null,
  "side": null,
  "price_band": {
    "range_low": "3150.00",
    "range_high": "3280.00"
  },
  "risk": {
    "max_position_fraction": "0.10",
    "suggested_quote_notional": null,
    "max_quote_notional": null
  },
  "guards": {
    "rsi14": "44.0",
    "atr14": "62.0",
    "buy_allowed": false,
    "sell_allowed": false,
    "no_loss_required": false,
    "no_loss_passed": null
  },
  "reason_codes": ["supporting_4h_downtrend_block", "hold_without_new_entry"]
}
```

## Alert Example

```json
{
  "intent_type": "alert",
  "execution_intent": "operator_alert",
  "strategy": "spot_grid",
  "regime": "risk_off",
  "symbol": "ETHUSDT",
  "timeframe": "1h",
  "reference_price": "3120.00",
  "target_price": null,
  "grid_level_index": null,
  "side": null,
  "price_band": {
    "range_low": "3090.00",
    "range_high": "3275.00"
  },
  "risk": {
    "max_position_fraction": "0.00",
    "suggested_quote_notional": null,
    "max_quote_notional": null
  },
  "guards": {
    "rsi14": "22.0",
    "atr14": "140.0",
    "buy_allowed": false,
    "sell_allowed": false,
    "no_loss_required": false,
    "no_loss_passed": null
  },
  "reason_codes": ["risk_off_alert", "manual_review_required"]
}
```

## Persisted Event Boundary

`bot_signal.persisted.v1` is a notification event, not the payload transport. It contains
metadata and `signal_id`, including `signal_key`, `run_id`, `instance_id`, `module_id`,
`symbol`, `timeframe`, `snapshot_id`, `signal_type`, `side`, `payload_schema`,
`payload_schema_version`, `payload_hash`, and optional `correlation_id`.

The event must not embed the full signal payload. Trade Execution must read the full JSON
payload from `_bot_platform.bot_signals.payload_json` by `signal_id` before validation or
execution.
