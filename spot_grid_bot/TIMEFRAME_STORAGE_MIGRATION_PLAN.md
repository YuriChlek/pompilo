# Timeframe Storage Migration Plan

## Goal

Bring candle storage and multi-timeframe analysis into a consistent design:

- `1h` candles are stored in dedicated `*_1h` tables
- `4h` candles are stored in dedicated `*_4h` tables
- sync writes both timeframes to separate tables
- runtime reads `1h` and `4h` from PostgreSQL directly
- `4h` is no longer derived in memory from `1h`

This plan changes architecture, storage naming, sync flow, runtime loading, tests, and documentation.

## Target Table Naming

Current suffix:

- `_p_candles`

Target suffixes:

- `_1h`
- `_4h`

Examples:

- `_candles_trading_data.ethusdt_1h`
- `_candles_trading_data.ethusdt_4h`

## Scope

Files that will be affected:

- `infrastructure/db.py`
- `infrastructure/binance_api.py`
- `infrastructure/binance_market_data_synchronizer.py`
- `infrastructure/market_data_provider.py`
- `domain/strategy_config.py`
- `application/bootstrap.py`
- `main.py`
- `tests/...` for sync, DB loading, planner context, runtime features
- `README.md`
- `README_UA.md`

## Implementation Phases

### Phase 1. Normalize storage naming

Goal:

- replace the ambiguous shared candle suffix with explicit timeframe-specific suffixes

Changes:

- introduce constants for:
  - `DEFAULT_CANDLE_TABLE_SUFFIX_1H = "_1h"`
  - `DEFAULT_CANDLE_TABLE_SUFFIX_4H = "_4h"`
- stop treating one generic candle table as the source of truth for every timeframe
- make table naming deterministic from `(symbol, timeframe)`

Expected result:

- table name resolution becomes explicit and timeframe-safe

## Phase 2. Update candle table creation

Goal:

- ensure both timeframe tables exist for every configured symbol

Changes:

- update `ensure_candle_tables(...)` in `infrastructure/db.py`
- create both:
  - `<symbol>_1h`
  - `<symbol>_4h`
- keep the same shared schema:
  - `_candles_trading_data`

Expected result:

- startup creates both required tables

## Phase 3. Make sync route by timeframe

Goal:

- write `1h` data to `*_1h`
- write `4h` data to `*_4h`

Changes:

- in `infrastructure/binance_api.py`, derive table suffix from the requested timeframe
- `fetch_and_store(...)` must resolve:
  - `1h -> _1h`
  - `4h -> _4h`
- reject unsupported timeframes explicitly instead of silently mixing them into the same storage path

Expected result:

- `python main.py sync --timeframe 1h` writes only to `*_1h`
- `python main.py sync --timeframe 4h` writes only to `*_4h`

## Phase 4. Sync both timeframes in the default sync flow

Goal:

- default operational sync must refresh both timeframes, not only `1h`

Changes:

- update `run_binance_candle_sync(...)`
- update `BinanceMarketDataSynchronizer`
- default scheduled sync should run:
  1. `1h` sync
  2. `4h` sync
- keep CLI support for targeted manual sync:
  - `--timeframe 1h`
  - `--timeframe 4h`

Expected result:

- live scheduled operation maintains both data sets continuously

## Phase 5. Add explicit repository loading for both timeframes

Goal:

- runtime should load real `1h` and real `4h` candles from DB

Changes:

- extend `DatabaseCandleRepository` to support timeframe-aware reads
- possible options:
  - one repository with `fetch_recent_candles(symbol, limit, timeframe)`
  - or two repository instances with different suffixes
- `DatabaseMarketDataProvider` should load:
  - base candles from `*_1h`
  - higher timeframe candles from `*_4h`

Expected result:

- no runtime aggregation from `1h` is needed anymore

## Phase 6. Remove in-memory 1h -> 4h aggregation

Goal:

- remove the temporary aggregation workaround introduced for higher timeframe confirmation

Changes:

- delete `_aggregate_candles(...)` from `infrastructure/market_data_provider.py`
- keep `MarketContext.higher_timeframe_candles`, but populate it from DB-backed `4h`

Expected result:

- higher timeframe analysis uses actual synced `4h` candles

## Phase 7. Align runtime config

Goal:

- make timeframe storage behavior explicit in config

Changes:

- in `domain/strategy_config.py`, split market-data naming/config from the old shared suffix model
- add explicit storage suffixes or timeframe mapping for:
  - `1h`
  - `4h`
- optionally keep the analysis timeframe and higher timeframe as named config values rather than hardcoded assumptions

Expected result:

- timeframe storage rules are visible in config instead of buried in infrastructure code

## Phase 8. Update CLI and operator behavior

Goal:

- make command behavior clear and non-misleading

Changes:

- `main.py sync --timeframe 1h` -> sync only `1h`
- `main.py sync --timeframe 4h` -> sync only `4h`
- document that `live` scheduled mode refreshes both timeframes automatically
- if needed, add a future `sync-all` behavior, but it is not required if scheduled sync already does both

Expected result:

- CLI matches actual storage semantics

## Phase 9. Data migration strategy

Goal:

- avoid ambiguous reuse of old `*_p_candles` tables

Changes:

- stop reading old `*_p_candles` tables in runtime
- create and populate fresh `*_1h` and `*_4h` tables
- if needed, provide a one-time migration script later, but this is optional if fresh re-sync is acceptable

Assumption:

- a clean re-sync is acceptable and simpler than trying to infer timeframe from old mixed tables

Expected result:

- no ambiguity about what data lives in each table

## Phase 10. Tests

Goal:

- cover the new storage contract directly

Test areas:

- `ensure_candle_tables(...)` creates both `_1h` and `_4h`
- `sync --timeframe 1h` resolves to `*_1h`
- `sync --timeframe 4h` resolves to `*_4h`
- scheduled sync runs both timeframes
- `DatabaseMarketDataProvider` loads `1h` and `4h` from separate sources
- `MarketContext.higher_timeframe_candles` contains DB-loaded `4h`
- multi-timeframe downtrend logic still blocks fresh buys correctly

Expected result:

- storage, sync, and planner integration are regression-safe

## Phase 11. Documentation

Goal:

- align docs with the real architecture

Changes:

- update `README.md`
- update `README_UA.md`
- explicitly document:
  - `*_1h` and `*_4h` table naming
  - scheduled dual-timeframe sync
  - targeted CLI sync per timeframe
  - higher timeframe now comes from DB, not aggregation

## Recommended Execution Order

1. Introduce explicit suffix constants and timeframe-to-table mapping.
2. Update table creation to create both `*_1h` and `*_4h`.
3. Route sync writes to timeframe-specific tables.
4. Make scheduled sync update both timeframes.
5. Update DB repository and market-data provider to read separate `1h` and `4h` tables.
6. Remove runtime aggregation logic.
7. Update tests.
8. Update README files.

## Acceptance Criteria

The migration is complete when all of the following are true:

- `1h` candles are stored only in `*_1h`
- `4h` candles are stored only in `*_4h`
- scheduled sync updates both timeframes
- runtime reads both timeframes from PostgreSQL
- no `1h -> 4h` aggregation remains in the market-data provider
- CLI behavior is explicit and documented
- tests pass with the new storage model
