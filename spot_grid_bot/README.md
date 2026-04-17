# Spot Grid Bot

## Overview

`spot_grid_bot` is an adaptive spot grid bot for Bybit demo trading.

It currently:

- creates and maintains its own candle schema/tables in PostgreSQL
- synchronizes candles directly from Binance Spot
- reads candles from PostgreSQL
- reads balances, open orders, ticker data, and executions from Bybit spot demo
- detects market regime using indicators and market structure
- evaluates risk and projected exposure
- sizes buy-side entries in `USDT` using percentage-based caps
- distributes new-entry budget through a portfolio-level allocator
- supports trigger-based underwater averaging from free `USDT` balance
- builds regime-aware grid or de-risk orders
- enforces no-loss sell behavior
- persists per-symbol runtime state in PostgreSQL
- includes a backtesting and evaluation layer with diagnostics

Entrypoint: [main.py](./main.py)

## Project Structure

- [main.py](./main.py): script entrypoint
- [application](./application): bootstrap, orchestration, scheduler, ports
- [domain](./domain): strategy, regime, risk, grid, inventory, cost basis, market structure
- [infrastructure](./infrastructure): Binance candle sync, Bybit exchange adapter, PostgreSQL candles, runtime state store
- [backtesting](./backtesting): backtest engine and reporting helpers
- [tests](./tests): unit and scenario behavior tests
- [utils](./utils): config, env loading, logging

## Runtime Flow

For each trading symbol, one cycle works like this:

1. Ensure candle schema and per-symbol candle tables exist.
2. In `live` mode, synchronize fresh candles from Binance Spot.
3. Load candles from PostgreSQL.
4. Load live balances and open orders from Bybit.
5. Restore persisted runtime state for the symbol.
6. Compute indicator snapshot.
7. Detect market regime.
8. Evaluate risk and projected exposure.
9. Run preliminary symbol analysis for all symbols.
10. Build one portfolio snapshot for the whole cycle.
11. Allocate portfolio-level entry budgets across eligible symbols.
12. Advance the state machine.
13. Build target orders.
14. Compare target orders to live orders.
15. If rebuild is required, sync orders to the exchange.
16. Persist updated runtime state.

Core orchestration: [application/trading_cycle_service.py](./application/trading_cycle_service.py)  
Planner: [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)
Portfolio allocator: [domain/portfolio_allocator.py](./domain/portfolio_allocator.py)

## Data Sources

### Candles

`spot_grid_bot` now owns the full candle-ingestion path.

Candles are fetched from Binance Spot and stored in PostgreSQL tables in the form:

- `_candles_trading_data.<symbol>_p_candles`

The candle schema matches `futures-trading-bot` exactly:

- `open_time`
- `close_time`
- `symbol`
- `open`
- `close`
- `high`
- `low`
- `cvd`
- `volume`
- `candle_id`

The bot:

- ensures the candle schema exists on startup
- ensures per-symbol candle tables exist on startup
- can run a standalone historical/refresh sync through CLI
- refreshes candles before each scheduled live cycle
- analyzes the latest `2400` hourly candles per symbol by default

Code:

- [infrastructure/db.py](./infrastructure/db.py)
- [infrastructure/binance_api.py](./infrastructure/binance_api.py)
- [infrastructure/binance_market_data_synchronizer.py](./infrastructure/binance_market_data_synchronizer.py)
- [infrastructure/market_data_provider.py](./infrastructure/market_data_provider.py)

### Live Exchange State

Balances, ticker, open orders, and execution history come from Bybit V5 demo spot API.

Code:

- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

## Market Regimes

The bot works with these regimes:

- `RANGE`
- `UPTREND`
- `DOWNTREND`
- `HIGH_VOLATILITY`
- `RISK_OFF`

The regime detector combines:

- `ema20`, `ema50`, `ema200`
- `ema50 slope`
- `atr14`
- realized volatility
- range width
- directional move
- abnormal candle and ATR spike flags
- market structure from swing highs and swing lows

Market structure layer:

- detects local swings
- falls back to segmented extrema when the trend is too smooth for classic pivots
- classifies structure as:
  - `BULLISH`
  - `BEARISH`
  - `RANGE`
  - `MIXED`
  - `NEUTRAL`

Trend regimes are now confirmed by both indicator context and structure:

- `UPTREND`: bullish EMA context plus bullish structure
- `DOWNTREND`: bearish EMA context plus bearish structure
- `RANGE`: flat/contained context or mixed/neutral structure
- broken or contradictory trend structure degrades toward `RANGE` instead of false trend continuation

Code:

- [domain/indicators.py](./domain/indicators.py)
- [domain/market_structure.py](./domain/market_structure.py)
- [domain/regime_detector.py](./domain/regime_detector.py)

## State Machine

The bot uses a state machine to avoid switching regime on every noisy bar.

It includes:

- hysteresis confirmation bars
- cooldown after state transition
- `volatility_cooldown_remaining`

Code:

- [domain/state_machine.py](./domain/state_machine.py)

In practice:

- one noisy candle should not instantly flip the regime
- after `HIGH_VOLATILITY`, entries can remain blocked for several bars

## Risk Logic

The risk layer evaluates:

- breakout kill switch
- abnormal volatility
- emergency volatility
- symbol inventory cap as `% of equity`
- symbol new-entry cap as `% of free quote`
- absolute symbol notional cap
- low quote reserve
- daily drawdown pause
- projected exposure from outstanding buy orders
- state-based restrictions in `DOWNTREND` and `RISK_OFF`

Code:

- [domain/risk_manager.py](./domain/risk_manager.py)
- [domain/exposure.py](./domain/exposure.py)

Risk output includes:

- `pause_entries`
- `force_risk_off`
- `cancel_entries`
- `allow_exit_only`
- `de_risk_mode = NONE | SOFT | HARD | PANIC`
- projected exposure diagnostics

Current sizing model:

- new buy-side budgets are calculated in `USDT`, not in raw coin units
- the planner uses the minimum of:
  - remaining symbol inventory room as `% of total equity`
  - symbol new-entry cap as `% of free quote`
  - global new-entry cap as `% of free quote`
  - absolute symbol notional cap
- if the resulting budget is below `min_symbol_entry_notional`, the bot skips new buy entries for that symbol

Code:

- [domain/allocation.py](./domain/allocation.py)
- [domain/inventory_manager.py](./domain/inventory_manager.py)
- [domain/risk_manager.py](./domain/risk_manager.py)

## Portfolio-Level Allocation

The bot now runs a portfolio-aware allocation pass before final order planning.

Current portfolio flow:

- collect preliminary analysis for all symbols
- build one shared portfolio snapshot
- compute one global new-entry budget from free quote and current outstanding exposure
- distribute that budget only across eligible symbols
- pass the portfolio budget cap down into the symbol-local allocation layer

Eligible symbols are currently limited to:

- `RANGE`
- `UPTREND`

Symbols are excluded from fresh entry budget when:

- `pause_entries=True`
- `force_risk_off=True`
- regime is `DOWNTREND`
- regime is `HIGH_VOLATILITY`
- regime is `RISK_OFF`

Budget distribution currently accounts for:

- regime weight
- regime confidence
- inventory pressure
- outstanding buy pressure
- projected quote-usage pressure
- underwater inventory penalty
- max concurrent entry symbols

This means:

- portfolio capital no longer depends only on symbol loop order
- only a bounded subset of symbols can receive fresh entry budget in one cycle
- underwater inventory is still penalized at the portfolio-allocation layer before any recovery logic is considered

## Venue-Aware Grid Handling

The grid is still ATR-based, but it is no longer planned as if every symbol shared the same price tick.

Current handling:

- raw grid geometry is built from ATR
- planner receives symbol-specific venue constraints from Bybit:
  - `tick_size`
  - `qty_step`
  - `min_order_qty`
  - `min_order_amt`
- planner applies a venue-viability pass before execution:
  - prices are normalized to symbol tick size
  - duplicate normalized levels are merged
  - sell ladders are reduced when current inventory cannot support the venue minimums
- execution then applies the live-price safety offset for buys

For cheap symbols such as `ENA`, `SUI`, `XRP`, and `DOGE`, this matters because:

- global `0.1` price rounding would collapse levels
- exchange minima can make multi-level sell ladders unrealistic
- moving the top buy level away from the market must preserve ladder structure instead of collapsing all buys into one price

Current buy-offset behavior:

- if the top buy level is too close to the live price, the whole buy ladder is shifted down
- the ladder shape is preserved instead of repricing each buy order to one identical level

Code:

- [domain/grid_builder.py](./domain/grid_builder.py)
- [domain/grid_viability.py](./domain/grid_viability.py)
- [domain/inventory_manager.py](./domain/inventory_manager.py)
- [infrastructure/market_data_provider.py](./infrastructure/market_data_provider.py)
- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

Code:

- [domain/portfolio_allocator.py](./domain/portfolio_allocator.py)
- [application/trading_cycle_service.py](./application/trading_cycle_service.py)

## Trading Logic

### RANGE

The bot builds a symmetric grid around the current price:

- buys below price
- sells above price
- width and step are ATR-based
- if range quality is weak, the bot can soften new buy entries instead of fully disabling trading
- if inventory is underwater and drawdown is above the recovery trigger, the bot can switch to a limited recovery buy budget

### UPTREND

The bot uses a separate pullback-and-scale-out policy:

- buy levels are placed below market as ATR-based pullbacks
- buy budget is weighted toward deeper pullback levels
- if price is too stretched above `ema20`, new trend buys are blocked
- if inventory is underwater and the recovery trigger is met, the bot can use a larger underwater-recovery budget than in `RANGE`
- sell levels are wider than in `RANGE`
- sell size increases at higher take-profit levels
- take-profit levels adapt to trend strength instead of always staying static

### DOWNTREND

New buys are blocked. The bot can enter staged de-risk behavior instead of opening new entries.

If the symbol already has open buy-entry orders from an earlier phase, the planner now forces a rebuild so those remaining entry buys are removed on the next sync.

### HIGH_VOLATILITY

New entries are suppressed and the bot uses explicit de-risk logic.

Open buy-entry orders are treated as stale for the current protective phase and are removed on rebuild.

### RISK_OFF

Protective mode:

- entries blocked
- de-risk path only
- remaining entry buys are cancelled on rebuild

Code:

- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)
- [domain/grid_builder.py](./domain/grid_builder.py)
- [domain/inventory_manager.py](./domain/inventory_manager.py)
- [domain/allocation.py](./domain/allocation.py)
- [domain/de_risk.py](./domain/de_risk.py)

This means:

- `BTC`, `SOL`, and `XRP` are sized through the same `USDT` risk model
- the bot is no longer dependent on one global `max_inventory_units` value for all assets
- cheaper and more expensive coins now scale through quote-currency budget instead of raw units
- symbol-local sizing now sits under a portfolio-level budget cap instead of operating fully independently

## No-Loss Sell Rule

The bot is intentionally constrained to avoid selling spot inventory at a loss.

Current behavior:

- it should not place sell orders below `cost_basis_price + 1%`
- if `cost_basis_price` is unavailable, it falls back to a conservative reference
- de-risk sells are also blocked if price is below the allowed exit floor

Code:

- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)
- [domain/de_risk.py](./domain/de_risk.py)
- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

This means:

- if inventory is underwater, the bot does not force a loss-taking exit
- it either holds inventory or continues operating through the appropriate phase
- if the market has already moved into a protective regime, fresh buy entries stay blocked until a valid recovery phase returns

## Underwater Averaging

The bot now supports a separate underwater-averaging path.

Current behavior:

- underwater averaging is disabled unless drawdown from `cost_basis_price` is above `UNDERWATER_AVERAGING_TRIGGER_PCT`
- averaging is only allowed in:
  - `RANGE`
  - `UPTREND`
- averaging is blocked in:
  - `DOWNTREND`
  - `HIGH_VOLATILITY`
  - `RISK_OFF`
- if drawdown exceeds `UNDERWATER_DEEP_STOP_PCT`, new underwater averaging is disabled again

Recovery budget is now calculated from free quote balance:

- base = current free `USDT` balance
- recovery budget = `free_quote * UNDERWATER_RECOVERY_BUDGET_PCT`
- then adjusted by regime:
  - `RANGE` uses `UNDERWATER_RANGE_BUDGET_MULTIPLIER`
  - `UPTREND` uses `UNDERWATER_UPTREND_BUDGET_MULTIPLIER`

Recovery sizing is still bounded by:

- remaining symbol inventory room
- actual free quote available
- venue minimum order constraints

This means:

- recovery averaging is no longer tied to a percentage of total equity
- on small accounts, recovery budget scales from free `USDT`, which is easier to reason about
- `RANGE` recovery is intentionally smaller than `UPTREND` recovery
- the bot does not average down indefinitely in bad regimes

## Cost Basis And Take-Profit Planning

`cost_basis_price` is derived from real Bybit spot executions:

- the bot reads recent fills
- calculates average cost for the remaining inventory
- caches the result briefly

Code:

- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

Phase 4 added cost-basis-aware sell planning:

- sell take-profit orders are not only filtered after planning
- they are now actively rebased from the effective inventory reference price
- minimum take-profit price is built from:
  - `cost_basis_price`
  - `min_sell_markup_bps`
  - ATR-based profit floor

Code:

- [domain/cost_basis.py](./domain/cost_basis.py)
- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)

This means:

- sell ladders are economically aligned with the actual spot inventory
- the bot no longer depends only on post-filtering to avoid bad exits
- underwater inventory is also penalized at the portfolio-allocation layer when deciding whether to resume accumulation
- if underwater averaging lowers `cost_basis_price`, future no-loss exits become easier to reach

## Rebuild Logic

The bot rebuilds not only on price movement, but also on material differences between:

- `target_orders`
- `live_orders`

Rebuild may also occur when:

- there are no live orders
- a state transition just happened
- `last_rebuild_price` is missing
- price deviation exceeds threshold
- target orders are empty while live orders still exist
- the symbol is in a protective regime and still has live buy-entry orders that must be cancelled

Code:

- [domain/order_diff.py](./domain/order_diff.py)
- [domain/spot_grid_planner.py](./domain/spot_grid_planner.py)

## Execution Logic

The Bybit spot demo executor:

- loads open orders
- compares them to target orders
- cancels unnecessary orders
- places missing orders
- respects exchange filters
- avoids duplicate `clientOrderId`

Guardrails include:

- `max_new_orders_per_cycle`
- `max_cancel_replace_per_cycle`
- `max_total_open_orders`
- `min_level_distance_bps`
- marketable order filter
- no-loss sell filter

Code:

- [infrastructure/execution_gateway.py](./infrastructure/execution_gateway.py)

Practical consequence:

- the strategy may plan more orders than the executor will place in one cycle
- execution is intentionally throttled for safety
- when the planner removes entry buys in a protective regime, the executor sync will cancel those stale buy orders on the venue

## Candle Ingestion And Independence

`spot_grid_bot` no longer has to rely on `futures-trading-bot` to populate candle history.

It now has its own:

- candle schema/table creation
- Binance Spot candle sync adapter
- scheduler hook for periodic candle refresh
- standalone `sync` CLI mode

This keeps the trading strategy unchanged while making the project operationally independent.

## Persistence

The bot persists runtime state per symbol in PostgreSQL:

- `regime`
- `bars_in_state`
- `cooldown_remaining`
- `volatility_cooldown_remaining`
- `pending_regime`
- `pending_count`
- `last_rebuild_price`
- `kill_switch_count`
- `recent_equity`

Code:

- [infrastructure/state_store.py](./infrastructure/state_store.py)
- [application/trading_cycle_service.py](./application/trading_cycle_service.py)

This means:

- after restart, the bot does not lose runtime memory
- risk state and state machine continue from persisted values
- one symbol does not share runtime state with another

## Backtesting And Evaluation

The project now includes an evaluation layer, not just a minimal backtest loop.

Backtest engine:

- simulates order fills
- updates inventory balances
- maintains simulated `cost_basis_price`
- tracks realized PnL on sells
- tracks unrealized PnL on remaining inventory
- counts rebuilds and de-risk events
- tracks blocked no-loss sell situations
- tracks risk reason frequency
- measures inventory utilization against the current symbol-level notional cap model

Code:

- [backtesting/engine.py](./backtesting/engine.py)

Backtest result metrics now include:

- `pnl`
- `realized_pnl`
- `unrealized_pnl`
- `max_drawdown`
- `trade_count`
- `rebuild_count`
- `average_inventory_utilization`
- `de_risk_event_count`
- `blocked_no_loss_sell_count`
- `risk_reason_counts`
- `regime_statistics`
- `kill_switch_count`

Reporting helpers:

- [backtesting/reporting.py](./backtesting/reporting.py)

Available helpers:

- `build_backtest_summary(...)`
- `format_backtest_summary(...)`

This gives the project a research/evaluation loop instead of only a live execution loop.

## Testing

The project includes behavior-focused tests for:

- rebuild idempotency
- execution guardrails
- no-loss de-risk behavior
- protective-regime cancellation of remaining buy-entry orders
- outstanding order exposure accounting
- percentage-based symbol allocation across low-priced and higher-priced assets
- portfolio-level budget distribution and concurrency limits
- underwater allocation penalty
- runtime state persistence and isolation
- cost-basis sell planning
- market-structure regime detection
- backtest diagnostics and reporting

Tests:

- [tests](./tests)

Run:

```bash
./.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Current Status

At the moment, the bot:

- runs as a live demo spot bot on Bybit
- owns its own Binance Spot candle synchronization path
- creates and maintains candle tables in `_candles_trading_data`
- reads candles from PostgreSQL
- enforces no-loss exit behavior
- uses explicit `HIGH_VOLATILITY` and staged de-risk handling
- accounts for outstanding buy-order exposure
- sizes new entries through percentage-based `USDT` budgets
- distributes fresh entry capital through a portfolio-level allocator
- cancels stale buy-entry orders when the symbol is already in a protective regime
- persists per-symbol runtime state
- builds cost-basis-aware take-profit sells
- uses a dedicated `UPTREND` policy:
  - ATR pullback buys
  - weighted deeper buy sizing
  - overextension entry block
  - staged sell-out with bigger size at higher prices
  - adaptive take-profit widening/narrowing with trend strength
- uses trigger-based underwater averaging:
  - recovery starts only after configured drawdown from `cost_basis_price`
  - recovery budget is calculated from free `USDT`
  - `RANGE` recovery is smaller than `UPTREND` recovery
  - `DOWNTREND` / `RISK_OFF` / `HIGH_VOLATILITY` do not allow averaging
- detects regime using both indicator context and market structure
- includes a diagnostics-oriented backtesting and reporting layer

## Run Commands

From the `spot_grid_bot` directory:

```bash
./.venv/bin/python main.py sync --period 365 --timeframe 1h
./.venv/bin/python main.py once
./.venv/bin/python main.py live
```
