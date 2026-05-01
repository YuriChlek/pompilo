# Spot Grid Bot

## Overview

`spot_grid_bot` is an adaptive spot grid bot for Bybit spot/demo trading with its own market-data ingestion, portfolio-aware planning, runtime state persistence, diagnostics, and operator tooling.

Current entrypoint: [main.py](./main.py)

The bot now includes:

- PostgreSQL-backed candle storage and runtime state persistence
- Binance Spot candle synchronization
- Bybit spot balances, live orders, and execution sync
- indicator + market-structure regime detection
- multi-timeframe regime confirmation
- risk-aware order planning and staged de-risking
- portfolio-level budget allocation across symbols
- persisted cost basis with `avgPrice` primary source and no-loss sell enforcement
- tick-aware order diffing and venue-aware grid normalization
- Telegram critical-event notifications
- live price WebSocket monitoring for off-cycle reaction
- health/state HTTP endpoints
- `dry-run` preview mode with structured diffs
- backtesting with diagnostics and HTML report export

## Project Structure

- [main.py](./main.py): CLI entrypoint and process mode dispatch
- [application](./application): bootstrap, scheduler, health, dry-run, orchestration services
- [domain](./domain): strategy, regime, risk, allocation, grid, cost basis, state machine
- [infrastructure](./infrastructure): Binance sync, Bybit adapters, PostgreSQL repositories, notifier, live price monitor
- [backtesting](./backtesting): backtest engine and reporting
- [tests](./tests): unit and scenario behavior tests
- [utils](./utils): environment loading, runtime config, logging

## Quick Start

### 1. Create the local environment

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

### 2. Configure the environment

Create or update `.env` with at least:

```env
DB_HOST=localhost
DB_PORT=5432
DATABASE=pompilo_db
DB_USER=admin
DB_PASS=admin_pass

BYBIT_API_KEY=...
BYBIT_API_SECRET=...
BYBIT_API_ENDPOINT=https://api-demo.bybit.com

SPOT_TRADING_SYMBOLS=ETHUSDT,SOLUSDT
EXECUTION_MODE=bybit_spot_demo
```

Optional operator features:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

HEALTHCHECK_HOST=0.0.0.0
HEALTHCHECK_PORT=8080

ENABLE_LIVE_PRICE_MONITOR=true
LIVE_PRICE_DEVIATION_ATR_MULTIPLIER=2.0
LIVE_PRICE_MONITOR_COOLDOWN_SECONDS=60
```

### 3. Sync candles

```bash
./.venv/bin/python main.py sync --period 365 --timeframe 1h
```

### 4. Run one planning/execution cycle

```bash
./.venv/bin/python main.py once
```

### 5. Preview changes without touching live orders

```bash
./.venv/bin/python main.py dry-run
```

### 6. Run the recurring live scheduler

```bash
./.venv/bin/python main.py live
```

### 7. Run the full test suite

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## CLI Commands

### `sync`

Downloads Binance Spot candles and stores them in PostgreSQL.

```bash
./.venv/bin/python main.py sync --period 365 --timeframe 1h
./.venv/bin/python main.py sync --period 30 --timeframe 4h
```

### `once`

Runs one full analysis/planning/execution cycle for all configured symbols.

```bash
./.venv/bin/python main.py once
```

### `dry-run`

Runs one full analysis/planning cycle and prints a structured diff against current live orders without syncing any order to the venue.

```bash
./.venv/bin/python main.py dry-run
```

Example output shape:

```text
[ETHUSDT] Regime: RANGE
[ETHUSDT] New BUY @ 1800.0 x 0.055
[ETHUSDT] Cancel BUY @ 1820.0 x 0.055
[ETHUSDT] Keep SELL @ 1950.0 x 0.03
```

### `live`

Runs the recurring scheduler. In live mode the process:

- waits for the configured schedule
- synchronizes candles before each scheduled cycle
- runs the trading cycle for all configured symbols
- exposes health endpoints
- optionally runs the WebSocket live-price monitor in the background

```bash
./.venv/bin/python main.py live
```

## Runtime Flow

For each symbol, the trading process now works like this:

1. Ensure candle schema and per-symbol candle tables exist.
2. Restore persisted per-symbol runtime state from PostgreSQL.
3. Synchronize candles from Binance before scheduled cycles.
4. Load recent 1h candles from PostgreSQL.
5. Build an aggregated higher-timeframe candle stream from the same base history.
6. Load live balances, mark price, instrument filters, and open orders from Bybit.
7. Resolve `cost_basis_price` from Bybit `avgPrice`, then fall back to persisted runtime state if needed.
8. Compute indicator snapshot.
9. Detect regime from indicators and market structure.
10. Apply higher-timeframe downtrend confirmation before final entry decisions.
11. Evaluate risk, projected exposure, and de-risk mode.
12. Run preliminary analysis for all symbols.
13. Build a portfolio snapshot and distribute symbol budgets.
14. Advance the symbol state machine.
15. Build target orders.
16. Apply no-loss, venue, and execution guardrails.
17. Diff target orders against current live orders.
18. Execute only when rebuild is required.
19. Persist updated runtime state back to PostgreSQL.
20. Update health/state snapshots for operator endpoints.

Core orchestration:

- [application/trading_cycle_service.py](./application/trading_cycle_service.py)
- [application/scheduler.py](./application/scheduler.py)
- [application/bootstrap.py](./application/bootstrap.py)

## Detailed Feature Set

### 1. Candle Ingestion and Storage

The bot owns the full candle-ingestion path.

- Candles are fetched from Binance Spot.
- Candles are stored in PostgreSQL under `_candles_trading_data.<symbol>_p_candles`.
- Candle schema/tables are created automatically when needed.
- The scheduler refreshes market data before each scheduled cycle.
- The planner works from PostgreSQL-backed history instead of direct exchange klines.

Relevant code:

- [infrastructure/db.py](./infrastructure/db.py)
- [infrastructure/binance_api.py](./infrastructure/binance_api.py)
- [infrastructure/binance_market_data_synchronizer.py](./infrastructure/binance_market_data_synchronizer.py)
- [infrastructure/market_data_provider.py](./infrastructure/market_data_provider.py)

### 2. Market Regime Detection

The strategy works with:

- `RANGE`
- `UPTREND`
- `DOWNTREND`
- `HIGH_VOLATILITY`
- `RISK_OFF`

Regime detection combines:

- `ema20`, `ema50`, `ema200`
- `ema50` slope
- `atr14`
- realized volatility
- range width
- directional move and directional sign
- abnormal candle and ATR spike flags
- swing-high / swing-low market structure

The detector now also uses higher-timeframe confirmation:

- the market-data provider builds a coarser candle stream from the base 1h history
- if the higher timeframe is in `DOWNTREND`, fresh entries are paused even when the lower timeframe still looks like `RANGE`

Relevant code:

- [domain/indicators.py](./domain/indicators.py)
- [domain/market_structure.py](./domain/market_structure.py)
- [domain/regime_detector.py](./domain/regime_detector.py)
- [domain/symbol_analyzer.py](./domain/symbol_analyzer.py)

### 3. State Machine and Runtime Memory

The bot keeps persistent per-symbol runtime state:

- current strategy state
- hysteresis/pending regime transition state
- volatility cooldown
- kill-switch history
- persisted `cost_basis_price`

The state machine is immutable on transitions, which makes preview analysis safe and avoids accidental state mutation during non-committing passes.

Relevant code:

- [domain/state_machine.py](./domain/state_machine.py)
- [domain/runtime_models.py](./domain/runtime_models.py)
- [infrastructure/state_store.py](./infrastructure/state_store.py)

### 4. Risk Logic

The risk layer evaluates:

- breakout kill switch
- abnormal volatility
- emergency volatility
- daily drawdown pause
- quote reserve pressure
- symbol-level and portfolio-level inventory limits
- projected outstanding buy exposure
- state-based restrictions in `DOWNTREND`, `HIGH_VOLATILITY`, and `RISK_OFF`

Risk output includes:

- `pause_entries`
- `force_risk_off`
- `cancel_entries`
- `allow_exit_only`
- `de_risk_mode = NONE | SOFT | HARD | PANIC`

Relevant code:

- [domain/risk_manager.py](./domain/risk_manager.py)
- [domain/exposure.py](./domain/exposure.py)

### 5. Portfolio-Level Allocation

The planner is portfolio-aware before final per-symbol order generation.

It now:

- runs preliminary symbol analysis across all configured symbols
- builds one shared portfolio snapshot
- computes one global new-entry budget from free quote and current outstanding exposure
- distributes that budget only across eligible symbols
- penalizes underwater inventory and exposure-heavy symbols before local grid sizing

Relevant code:

- [domain/portfolio_allocator.py](./domain/portfolio_allocator.py)
- [domain/allocation.py](./domain/allocation.py)
- [application/analysis_batch_service.py](./application/analysis_batch_service.py)

### 6. Cost Basis and No-Loss Sell Enforcement

The bot now treats Bybit `avgPrice` as the primary source of spot cost basis.

Behavior:

- primary source: Bybit wallet `avgPrice`
- fallback source: persisted `cost_basis_price` from PostgreSQL/runtime state
- if both are unavailable and inventory is open, `SELL` planning is blocked
- when inventory reaches zero, persisted `cost_basis_price` is cleared

No-loss logic:

- sell targets are rebased above the minimum exit floor
- sell orders below the no-loss floor are blocked
- planning logs when sell levels are removed because cost basis is missing or the no-loss floor rejects them

Relevant code:

- [infrastructure/bybit_account_client.py](./infrastructure/bybit_account_client.py)
- [domain/cost_basis.py](./domain/cost_basis.py)
- [domain/target_order_builder.py](./domain/target_order_builder.py)
- [infrastructure/execution_guardrails.py](./infrastructure/execution_guardrails.py)

### 7. Grid Planning and Venue Awareness

Grid planning is now venue-aware.

The bot:

- builds ATR-based range and trend grids
- respects symbol-specific tick size and size step
- merges duplicate normalized levels after tick rounding
- preserves merged source tags for diagnostics
- avoids false target/live mismatches on cheap symbols because one-tick drift is treated safely

Relevant code:

- [domain/grid_builder.py](./domain/grid_builder.py)
- [domain/grid_viability.py](./domain/grid_viability.py)
- [domain/order_diff.py](./domain/order_diff.py)

### 8. Underwater Recovery Logic

When inventory is underwater, the strategy can shift into controlled recovery behavior.

The bot can:

- reduce or cap new buy levels
- limit recovery budget to a fraction of free quote
- use different recovery budgets for `RANGE` and `UPTREND`
- tighten sell aggressiveness around recovery exits

Relevant code:

- [domain/uptrend_policy.py](./domain/uptrend_policy.py)
- [domain/target_order_builder.py](./domain/target_order_builder.py)

### 9. Execution Guardrails

Before syncing target orders to the venue, the bot applies:

- no-loss sell filtering
- marketable-order filtering
- near-duplicate level deduplication
- per-cycle order throttles
- venue minimum notional / minimum quantity normalization

Guardrails are intentionally applied in domain-first order:

1. no-loss restrictions
2. marketability checks
3. dedupe / cleanup
4. execution throttles

Relevant code:

- [infrastructure/execution_guardrails.py](./infrastructure/execution_guardrails.py)
- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

### 10. Notifications

The notifier layer currently supports:

- logging-only notifications by default
- Telegram notifications for critical rebuild/risk events

Telegram alerts are sent only for high-severity cases such as:

- `force_risk_off`
- `de_risk_mode = HARD | PANIC`
- kill-switch-triggered behavior
- critical risk reasons such as `daily_drawdown_pause` or `emergency_volatility`

Relevant code:

- [infrastructure/notifications.py](./infrastructure/notifications.py)

### 11. Live Price Monitoring

In `live` mode, the scheduler can run a Bybit public WebSocket monitor between regular cycles.

Behavior:

- subscribes to ticker updates
- compares live price vs cached cycle reference
- uses ATR-based deviation thresholds
- triggers an off-cycle symbol-only trading pass on large price shocks
- throttles repeated alerts with a cooldown

Relevant code:

- [infrastructure/live_price_monitor.py](./infrastructure/live_price_monitor.py)
- [application/scheduler.py](./application/scheduler.py)

### 12. Health and State Endpoints

When `HEALTHCHECK_PORT > 0`, the live process exposes:

- `GET /health`
- `GET /state`

`/health` returns process-level liveness information:

```json
{
  "status": "ok",
  "last_cycle": "2026-05-01T15:00:01+00:00",
  "symbols": ["ETHUSDT", "SOLUSDT"]
}
```

`/state` returns the latest in-memory runtime summary per symbol, including current regime and kill-switch count.

Relevant code:

- [application/health.py](./application/health.py)

### 13. Dry-Run Preview Mode

`dry-run` executes the full planning path but never syncs orders.

It is useful for:

- safe debugging against the real venue state
- reviewing what would be created, kept, or cancelled
- inspecting regime/risk-driven diffs before enabling live execution

Relevant code:

- [application/dry_run.py](./application/dry_run.py)
- [main.py](./main.py)

### 14. Backtesting and Reporting

The backtest layer now includes:

- historical order simulation
- slippage-aware fill accounting
- optional maker-fee accounting
- no-loss sell diagnostics
- rebuild count and de-risk diagnostics
- risk-reason frequency tracking
- final inventory and unrealized/realized PnL reporting
- HTML report export

Programmatic usage:

```python
from backtesting import BacktestEngine, export_html_report
from domain.strategy_config import DEFAULT_STRATEGY_CONFIG

engine = BacktestEngine(DEFAULT_STRATEGY_CONFIG)
result = engine.run("ETHUSDT", candles)
export_html_report(result, "backtest_report.html")
```

Relevant code:

- [backtesting/engine.py](./backtesting/engine.py)
- [backtesting/reporting.py](./backtesting/reporting.py)

## Key Runtime Configuration

Important environment variables:

- `SPOT_TRADING_SYMBOLS`: comma-separated symbol list
- `EXECUTION_MODE`: execution adapter mode
- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `BYBIT_API_ENDPOINT`
- `DB_HOST`
- `DB_PORT`
- `DATABASE`
- `DB_USER`
- `DB_PASS`
- `CANDLE_LOOKBACK`
- `MAX_NEW_ORDERS_PER_CYCLE`
- `MIN_SYMBOL_ENTRY_NOTIONAL`
- `MAX_SYMBOL_INVENTORY_PCT_OF_EQUITY`
- `MAX_SYMBOL_NEW_ENTRY_PCT_OF_FREE_QUOTE`
- `UNDERWATER_AVERAGING_ENABLED`
- `UNDERWATER_AVERAGING_TRIGGER_PCT`
- `UNDERWATER_RECOVERY_BUDGET_PCT`
- `UNDERWATER_RANGE_BUDGET_MULTIPLIER`
- `UNDERWATER_UPTREND_BUDGET_MULTIPLIER`
- `UNDERWATER_MAX_RECOVERY_LEVELS`
- `UNDERWATER_DEEP_STOP_PCT`
- `RUN_TARGET_MINUTE`
- `RUN_TARGET_SECOND`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`
- `HEALTHCHECK_HOST`
- `HEALTHCHECK_PORT`
- `ENABLE_LIVE_PRICE_MONITOR`
- `LIVE_PRICE_DEVIATION_ATR_MULTIPLIER`
- `LIVE_PRICE_MONITOR_COOLDOWN_SECONDS`

Source of truth:

- [utils/config.py](./utils/config.py)

## Testing

The test suite covers:

- regime detection and state transitions
- runtime state persistence
- cost basis handling and no-loss sell logic
- execution guardrails
- portfolio allocation
- underwater recovery logic
- cheap-symbol tick handling
- scheduler resilience
- Telegram notifier behavior
- health endpoint behavior
- live price monitor deviation handling
- dry-run formatting
- backtest diagnostics and reporting

Run:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Current suite status:

- `94` tests passing locally in `.venv`

