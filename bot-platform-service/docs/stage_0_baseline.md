# Stage 0 Baseline and Boundaries

## Purpose

This document records the current standalone behavior of `spot_grid_bot` and `spot-greenwich-bot` before Bot Platform integration work starts.

Stage 0 does not change either legacy bot runtime. It only fixes the baseline, names the existing boundaries, and adds smoke coverage that platform core work must not import concrete bots directly.

Implementation directory for the platform remains:

```text
bot-platform-service/
```

Concrete bot adapters, when introduced later, must live under:

```text
bot-platform-service/src/bot_platform_service/infrastructure/bot_modules/
```

## Baseline Summary

### `spot_grid_bot`

Current role:

- adaptive spot grid bot for Bybit spot/demo trading;
- owns standalone candle synchronization from Binance into PostgreSQL;
- owns standalone execution sync against Bybit or paper execution;
- produces strategy decisions with target grid orders and rebuild decisions.

Current CLI modes:

- `sync --period <days> --timeframe 1h|4h`
  - creates candle tables;
  - downloads Binance Spot candles;
  - stores candles in the bot-owned candle schema.
- `dry-run`
  - initializes the live trading cycle;
  - restores state and reconciles configured dependencies;
  - previews target-vs-live order diffs;
  - does not sync orders.
- `live`
  - starts recurring scheduler;
  - refreshes 1h and 4h candles before scheduled cycles;
  - runs the trading cycle;
  - exposes health endpoints when configured;
  - can run a live price monitor when enabled and not in paper mode.

Application ports:

- `MarketDataProvider`
  - returns full `MarketContext`;
  - current context includes candles, balances, live orders, inventory, mark price, venue constraints, and cost basis inputs.
- `OrderExecutor`
  - reconciles execution state;
  - syncs target orders to the external venue.
- `SignalNotifier`
  - sends notifications about rebuild/trading decisions.
- `MarketDataSynchronizer`
  - refreshes external market data on schedule.
- `StateStore`
  - initializes, loads, and saves per-symbol runtime state.

Market-data dependencies:

- standalone `sync` path uses Binance Spot candle fetching;
- live scheduler uses `BinanceMarketDataSynchronizer`;
- trading cycle uses `DatabaseMarketDataProvider`;
- candle data is stored in PostgreSQL tables owned by the bot runtime;
- platform mode must replace this with a platform-provided `MarketDataSnapshotProvider` backed by Market Data Service snapshots.

Execution and private exchange dependencies:

- `build_live_trading_cycle()` constructs `BybitSpotExecutionService` unless `EXECUTION_MODE == "paper"`;
- `PaperExecutionService` is the local non-venue alternative;
- `DatabaseMarketDataProvider` is currently constructed with `executor.exchange`;
- live price monitoring uses Bybit public WebSocket data;
- platform mode must not construct `BybitSpotExecutionService`, private Bybit account clients, or any venue order-sync path.

Signal-generation dependencies:

- primary planner is `SpotGridPlanner`;
- market analysis uses indicator snapshots, regime detection, risk decisions, portfolio allocation, inventory rules, and target order building;
- current output is `StrategyDecision`, including `target_orders`, `live_orders`, `rebuild_required`, `regime`, risk data, and reasons;
- platform adapter must translate these decisions into standardized `BotSignal` records, not order intents.

Standalone runtime flow remains unchanged at Stage 0.

## `spot-greenwich-bot`

Current role:

- signal-driven spot execution bot using Greenwich mean reversion;
- owns standalone D1 and H4 candle synchronization from Binance into PostgreSQL;
- owns standalone position reconciliation and market order execution on Bybit;
- supports dry-run and notification-only behavior.

Current CLI modes:

- `sync --period <days>`
  - syncs D1 candles from Binance.
- `sync-3y`
  - syncs 3 years of D1 candles.
- `sync-4h --period <days>`
  - syncs H4 candles from Binance.
- `sync-full`
  - syncs both D1 and H4 candles.
- `analyze --symbol <symbol> --timeframe 4h --dry-run`
  - runs one analysis cycle without real exchange orders.
- `analyze --symbol <symbol> --timeframe 4h --notification-only`
  - enters scheduler-like notification-only runtime for selected symbols.
- `init-db`
  - creates candle and spot-ledger tables.
- `migrate`
  - runs SQL migrations.
- default scheduled runtime with root `--dry-run` and `--notification-only` flags
  - starts the live scheduler through `RuntimeCommandService.live()`.

Application ports:

- `MarketDataProvider`
  - returns symbol candle history.
- `MarketDataSynchronizer`
  - refreshes D1/H4 market data before a cycle.
- `PositionExecutor`
  - reads reconciled position state;
  - reads quote balance;
  - executes one decision.
- `SignalNotifier`
  - publishes notification output for processed signals.
- `StateStore`
  - optional lightweight symbol runtime state persistence.

Market-data dependencies:

- standalone sync path uses `api.run_api()` and `BinanceAPI`;
- live scheduler uses `BinanceMarketDataSynchronizer`;
- trading cycle uses `MultiTimeframeMarketDataProvider`;
- runtime expects D1 and H4 candle tables;
- platform mode must replace this with a platform-provided `MarketDataSnapshotProvider` for canonical `1d` and `4h` snapshots.

Execution and private exchange dependencies:

- `build_live_trading_cycle()` constructs `BybitSpotExecutor`;
- `build_initialization_service()` also constructs `BybitSpotExecutor` for startup reconciliation;
- `PositionExecutor.execute()` may place or simulate Bybit spot orders depending on `dry_run` and `notification_only_mode`;
- notification-only is currently implemented through execution-service behavior, so platform mode must route notifications through platform `NotificationPublisher` instead;
- platform mode must not construct `BybitSpotExecutor`, `BybitSpotClient`, or any private exchange execution path.

Signal-generation dependencies:

- raw signal generation lives in `domain.signals.generate_spot_signal()`;
- `GreenwichSpotPlanner` combines raw signal generation with execution policy;
- `MultiTimeframeSpotPlanner` applies D1 regime filtering over H4 signals;
- current output includes `SpotSignal`, `ExecutionDecision`, `ExecutionResult`, and `PositionState`;
- platform adapter must translate `SpotSignal` and execution decisions into standardized `BotSignal` records, not order intents.

Standalone runtime flow remains unchanged at Stage 0.

## Stage 0 Boundary Decisions

- Bot Platform core must not import `spot_grid_bot` or `spot-greenwich-bot` directly.
- Later concrete bot imports are allowed only inside `bot_platform_service/infrastructure/bot_modules/`.
- Bot Platform remains signal-only.
- Bot Platform must not create exchange orders or private exchange clients.
- Bot Platform must not run Binance candle sync or read legacy candle tables in platform mode.
- Market data for platform mode must come from Market Data Service snapshots.
- Current standalone CLI behavior of both bots is preserved as rollback until migration rollout proves platform mode.

## Stage 0 Completion Evidence

- This document records CLI modes, application ports, market-data dependencies, execution/private exchange dependencies, and signal-generation dependencies for both bots.
- Smoke tests under `bot-platform-service/tests/smoke/` verify that platform package code does not import concrete legacy bots directly outside the future adapter directory.
