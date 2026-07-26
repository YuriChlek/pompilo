# Spot Grid Indicator Library Decision

This document records the phase 8 decision for platform-native `spot_grid` indicator
calculation.

## Decision

Use `stock-indicators` as the selected indicator library for platform-native `spot_grid`.

Do not use package-local hand-rolled implementations as the target source of truth for
EMA, ATR, RSI, or volatility-adjacent indicators when `stock-indicators` supports the
calculation. Any later fallback path must be proposed explicitly, covered by parity tests
against `stock-indicators`, and have a documented removal condition before it is merged.

The dependency is declared in `bot-platform-service/pyproject.toml`:

```toml
"stock-indicators>=1.3"
```

Do not add `pandas`, `pandas-ta`, or `ta` for the initial Spot Grid migration.

## Rationale

- `stock-indicators` provides maintained implementations for EMA, ATR, RSI, and a broad
  set of time-series indicators.
- Its Python `Quote` boundary accepts OHLCV input and stores values as Decimal-compatible
  quote data before invoking the indicator functions.
- The library exposes explicit result series semantics and warmup behavior; tests must
  lock the selected output values used by `spot_grid`.
- Using one selected package prevents the module from drifting into incompatible custom
  formulas.

## Boundary Rules

- `manifest.py`, `config_schema.py`, and `bot_config.py` must stay lightweight-import safe.
- Do not import `stock_indicators`, `pandas`, `pandas_ta`, `ta`, or NumPy from those
  lightweight files.
- Keep `stock-indicators` result objects at the indicator adapter/calculator boundary only.
- Normalize every library output into `Decimal` before constructing domain-facing DTOs.
- Float values are not allowed in domain-facing trading DTOs. External non-trading ratios
  may be accepted only at the library boundary and immediately converted with
  `normalize_indicator_decimal(..., allow_external_float=True)`.
- Exact fixture tests must fail when a dependency upgrade changes EMA/ATR/RSI/volatility
  outputs without an intentional expected-value update.

## Follow-Up

Phase 9 adapts platform candles to `stock-indicators` quote input. Phases 10 and 11 lock
EMA, ATR, RSI, realized volatility, and volume outputs with fixture tests.
