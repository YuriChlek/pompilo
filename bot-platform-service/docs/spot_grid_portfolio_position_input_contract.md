# Spot Grid Portfolio And Position Input Contract

This contract defines the platform input used by `spot_grid` when strategy planning needs
portfolio or position context.

## Ownership

- `spot_grid` consumes `PortfolioContext` and `PositionContext` as explicit platform
  inputs.
- The bot module must not query private exchanges, account APIs, wallets, balances, fills,
  or orders directly.
- Platform orchestration, or another platform-owned adapter outside `spot_grid`, is
  responsible for resolving account and portfolio data into this execution-neutral input.
- The application boundary is `SpotGridPortfolioContextProvider`.

## Provider Port

`SpotGridPortfolioContextProvider.get_portfolio_context(...)` accepts:

- `instance_id`
- `symbol`
- `timeframe`
- `snapshot_id`

It returns `PortfolioContext` or `None`.

Returning `None` means portfolio input is unavailable. The strategy must continue in
conservative mode rather than calling an exchange client.

## PortfolioContext

Fields:

- `total_equity`: Decimal portfolio equity represented in the strategy quote basis.
- `available_quote`: Decimal quote balance available to the strategy.
- `positions`: tuple of `PositionContext`.

`PortfolioContext` is JSON-safe through `to_payload()`, with Decimal values serialized as
strings.

## PositionContext

Fields:

- `symbol`: canonical symbol such as `ETHUSDT`.
- `base_quantity`: Decimal base asset quantity.
- `quote_notional`: Decimal quote notional for the position.
- `cost_basis`: optional Decimal average cost basis.
- `min_no_loss_exit_price`: optional Decimal platform-provided no-loss exit threshold.

`PositionContext` must not contain exchange order ids, fill ids, private account ids,
secrets, or order lifecycle state.

## Conservative Empty Fallback

When portfolio input is missing, `spot_grid` uses:

```json
{
  "total_equity": "0",
  "available_quote": "0",
  "positions": []
}
```

This fallback keeps the bot signal-only and conservative:

- no direct exchange query is attempted;
- ordinary sell intents have no cost basis and remain blocked by no-loss policy;
- diagnostics expose `portfolio_context.fallback = "empty_conservative"`.
