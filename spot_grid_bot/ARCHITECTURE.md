# Spot Grid Bot Architecture

## Goal

`spot_grid_bot` is a multi-symbol spot trading bot that:

- syncs candle data from Binance Spot into PostgreSQL
- reads balances and open orders from Bybit Spot
- detects market regime per symbol
- allocates entry budget at portfolio level
- builds symbol-level target orders
- synchronizes those orders to the exchange

The project is organized into `application`, `domain`, `infrastructure`, and `backtesting` layers.

## Layer Overview

### `application/`

Application orchestration lives here.

- [trading_cycle_service.py](./application/trading_cycle_service.py)
  - main facade for `initialize`, `run`, `run_many`
- [initialization_service.py](./application/initialization_service.py)
  - startup initialization
  - candle-table ensure
  - exchange reconcile
  - runtime-state restore
- [analysis_batch_service.py](./application/analysis_batch_service.py)
  - first pass over all symbols
  - builds analyses and portfolio allocation inputs
- [execution_service.py](./application/execution_service.py)
  - second pass
  - planner execution
  - persistence
  - notifications
- [bootstrap.py](./application/bootstrap.py)
  - runtime wiring
- [scheduler.py](./application/scheduler.py)
  - periodic live execution
- [ports.py](./application/ports.py)
  - application-facing protocols

### `domain/`

Trading logic lives here.

Core planning:

- [spot_grid_planner.py](./domain/spot_grid_planner.py)
  - main planner facade
- [symbol_analyzer.py](./domain/symbol_analyzer.py)
  - first-pass symbol analysis
- [target_order_builder.py](./domain/target_order_builder.py)
  - builds final target orders
- [rebuild_policy.py](./domain/rebuild_policy.py)
  - rebuild decision rules
- [live_order_policy.py](./domain/live_order_policy.py)
  - bot-managed live-order interpretation

Market and regime:

- [indicators.py](./domain/indicators.py)
- [market_structure.py](./domain/market_structure.py)
- [regime_detector.py](./domain/regime_detector.py)
- [state_machine.py](./domain/state_machine.py)

Risk and execution intent:

- [risk_manager.py](./domain/risk_manager.py)
- [exposure.py](./domain/exposure.py)
- [de_risk.py](./domain/de_risk.py)
- [cost_basis.py](./domain/cost_basis.py)

Grid and sizing:

- [grid_builder.py](./domain/grid_builder.py)
- [grid_viability.py](./domain/grid_viability.py)
- [inventory_manager.py](./domain/inventory_manager.py)
- [allocation.py](./domain/allocation.py)
- [portfolio_allocator.py](./domain/portfolio_allocator.py)
- [uptrend_policy.py](./domain/uptrend_policy.py)

Domain models:

- [models.py](./domain/models.py)
  - compatibility re-export layer
- [market_models.py](./domain/market_models.py)
- [order_models.py](./domain/order_models.py)
- [risk_models.py](./domain/risk_models.py)
- [runtime_models.py](./domain/runtime_models.py)
- [portfolio_models.py](./domain/portfolio_models.py)
- [inventory_models.py](./domain/inventory_models.py)
- [strategy_models.py](./domain/strategy_models.py)
- [backtest_models.py](./domain/backtest_models.py)

### `infrastructure/`

External systems and adapters live here.

Market data:

- [binance_api.py](./infrastructure/binance_api.py)
  - Binance Spot candle fetch and storage
- [binance_market_data_synchronizer.py](./infrastructure/binance_market_data_synchronizer.py)
  - scheduled candle refresh adapter
- [market_data_provider.py](./infrastructure/market_data_provider.py)
  - builds `MarketContext` from PostgreSQL + exchange state
- [db.py](./infrastructure/db.py)
  - candle schema/table helpers
  - candle repository
- by default, the planner uses the latest `2400` hourly candles per symbol

Exchange execution:

- [execution_gateway.py](./infrastructure/execution_gateway.py)
  - live and paper execution facade
- [bybit_market_client.py](./infrastructure/bybit_market_client.py)
  - prices and instrument filters
- [bybit_account_client.py](./infrastructure/bybit_account_client.py)
  - balances and account-derived inventory state
- [bybit_spot_types.py](./infrastructure/bybit_spot_types.py)
  - shared Bybit spot types and helpers
- [execution_guardrails.py](./infrastructure/execution_guardrails.py)
  - final order safety filters
- [cost_basis_resolver.py](./infrastructure/cost_basis_resolver.py)
  - cost basis calculation from executions

Persistence and reporting:

- [state_store.py](./infrastructure/state_store.py)
  - runtime-state persistence
- [notifications.py](./infrastructure/notifications.py)
  - logging notifier

### `backtesting/`

- [engine.py](./backtesting/engine.py)
  - offline strategy evaluation using the same planner stack

## Runtime Flow

### Live mode

1. Scheduler waits for the configured run time.
2. Binance candle sync updates PostgreSQL candle tables.
3. Trading cycle initializes runtime dependencies if needed.
4. First pass analyzes all symbols using the latest `2400` hourly candles per symbol by default.
5. Portfolio allocator distributes entry budget.
6. Second pass builds target orders per symbol.
7. Execution gateway synchronizes guarded orders to Bybit.
8. Runtime state is persisted.

### Once mode

1. Ensure candle tables and runtime tables.
2. Load latest candles from PostgreSQL.
3. Run the same two-pass planning flow.
4. Synchronize orders once.

### Sync mode

1. Ensure candle schema and symbol tables.
2. Pull candles from Binance Spot.
3. Upsert them into `_candles_trading_data.<symbol>_p_candles`.

## Design Principles

- Domain logic should stay independent from exchange SDK details.
- Planner behavior should not depend on raw venue quirks until the viability and execution layers.
- Infrastructure modules should have narrow responsibilities.
- Refactors should preserve strategy behavior and public runtime commands.

## Naming Convention

The project now follows these naming patterns:

- `*_service.py`
  - application orchestration
- `*_planner.py`
  - top-level planning facade
- `*_builder.py`
  - grid or target construction
- `*_manager.py`
  - domain logic manager
- `*_detector.py`
  - detection logic
- `*_policy.py`
  - isolated policy rules
- `*_client.py`
  - provider-specific infrastructure client
- `*_provider.py`
  - context/data provider
- `*_gateway.py`
  - exchange execution facade
- `*_synchronizer.py`
  - scheduled refresh adapter
- `*_models.py`
  - grouped DTO/model definitions
