# Timeframe Storage Cutover

## Purpose

This project no longer treats legacy `*_p_candles` tables as a valid runtime source for candle data.

Canonical runtime tables are now:

- `_candles_trading_data.<symbol>_1h`
- `_candles_trading_data.<symbol>_4h`

## Cutover Rule

Runtime and sync logic now assume:

- `1h` data lives only in `*_1h`
- `4h` data lives only in `*_4h`
- legacy `*_p_candles` tables are not read by runtime code

## Operational Migration

Recommended cutover path:

1. Deploy the new code.
2. Ensure new `*_1h` and `*_4h` tables are created.
3. Run a fresh manual sync into the new tables:

```bash
./.venv/bin/python main.py sync --period 365 --timeframe 1h
./.venv/bin/python main.py sync --period 365 --timeframe 4h
```

4. Start normal `live` operation.

## Why no automatic legacy migration

The old `*_p_candles` naming does not safely encode timeframe semantics.

If a table had been reused for multiple intervals in the past, the code cannot infer whether rows should be treated as `1h` or `4h` without operator knowledge.

Because of that, the safe migration path is:

- create fresh canonical tables
- re-sync both timeframes from Binance
- ignore old mixed/ambiguous storage
