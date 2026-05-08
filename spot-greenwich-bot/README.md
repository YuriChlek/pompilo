# Spot Greenwich Bot

## Overview

`spot-greenwich-bot` is a spot trading bot with a clean `application / domain / infrastructure` architecture.

The current runtime does the following:

- syncs `D1` and `4H` candles from Binance into PostgreSQL
- uses `D1` as a market-regime filter
- uses `4H` as the working execution timeframe
- generates Greenwich-based `BUY` and `SELL` signals
- reconciles local runtime state against Bybit balances and trade history
- executes spot market orders on Bybit
- stores runtime state and execution history in PostgreSQL
- sends logging and Telegram notifications
- supports `dry-run` and `notification-only` modes

This is not a grid bot and not a multi-order book-making system. It is a signal-driven spot execution bot.

## Strategy Summary

The core strategy is Greenwich mean reversion:

- `BUY`: recovery back above `lower3`
- `SELL`: crossunder below `upper2`

The current runtime adds several execution and filtering rules on top of the base Greenwich signal:

- `D1` regime filter: `death cross` + `ADX >= 30` blocks new `BUY` on `4H`
- confirmation candle for `BUY`
- anti-crash block for sharp recent drops
- averaging limit with reduced sizing on later entries
- partial take-profit at `upper1`
- ATR-normalized `BUY` sizing
- `BUY` market price guard
- portfolio position cap with priority symbols
- no-loss guard for `SELL`

The implementation details live primarily in:

- [domain/signals.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/signals.py:1)
- [domain/planner.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/planner.py:1)
- [domain/execution.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/execution.py:1)
- [infrastructure/execution_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/execution_service.py:1)

## Strategy Rules

### When the bot enters a position

The bot opens or adds to a spot position only when all of the following are true:

1. The base Greenwich `BUY` setup exists on `4H`.
   - The raw idea is a recovery from an oversold state around `lower3`.
2. The `D1` regime filter allows new `BUY`.
   - If `WMA(50) < WMA(200)` and `ADX >= 30` on `D1`, new `BUY` is blocked.
3. The confirmation candle logic allows the entry.
   - A fresh `BUY` signal is not entered immediately.
   - The next closed candle must confirm the recovery by closing above `lower3`.
4. The anti-crash filter passes.
   - If the recent drop over the configured lookback is too large, the `BUY` is blocked.
5. The execution policy allows the entry.
   - For an existing position, the new `BUY` price must be better than the current average entry price.
   - The averaging limit must not be exceeded.
   - Portfolio cap must allow opening a new symbol if this would be a fresh position.
6. The infrastructure price guard allows the market order.
   - If current market price is too far above the signal price, the `BUY` is blocked.

### How averaging works

The bot can add to an existing position, but only in a controlled way:

- first entry: `100%` of base size
- second entry: `60%` of base size
- third entry: `30%` of base size
- fourth and later entries: blocked

This means the bot does not keep averaging indefinitely in a downtrend.

### When the bot exits a position

The bot has two exit paths:

1. Partial take-profit:
   - if a position is open and `upper1` is reached, the bot sells `50%`
   - this can happen only once per position
2. Final exit:
   - when the normal Greenwich `SELL` signal appears, the bot sells the remaining position
   - the raw `SELL` condition is a crossunder below `upper2`

### When the bot does not trade

The bot intentionally skips trading in these cases:

- there is no valid Greenwich signal
- `D1` regime blocks new `BUY`
- the confirmation candle has not appeared yet
- the confirmation candle failed
- the anti-crash filter blocks the entry
- a new `BUY` is not better than the current average entry price
- the averaging entry limit is already reached
- portfolio cap blocks opening another symbol
- market price deviates too far above the `BUY` signal price
- normalized order quantity is below Bybit minimum filters
- `SELL` is not profitable enough under `MIN_PROFIT_RATIO`
- the no-loss guard sees that the live sell price has already dropped below the minimum profitable threshold
- the same signal was already executed and is treated as duplicate

In short, the bot is selective by design. It prefers to skip a trade rather than force one when the setup quality or execution quality is poor.

## Runtime Flow

One trading cycle works like this:

1. Load candle history from PostgreSQL.
2. Load and reconcile the current position state from Bybit and local DB state.
3. Build a trading plan:
   - raw Greenwich signal
   - signal filters
   - execution decision
4. Apply portfolio-level admission rules in multi-symbol runs.
5. Execute the decision through the Bybit executor, or simulate it in `dry-run` / `notification-only`.
6. Persist runtime state and order ledger updates.
7. Send notifications.

Main composition and orchestration entrypoints:

- [main.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/main.py:1)
- [application/bootstrap.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/bootstrap.py:1)
- [application/runtime_commands.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/runtime_commands.py:1)
- [application/trading_cycle_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/trading_cycle_service.py:1)
- [application/scheduler.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/scheduler.py:1)

## Architecture

### Application layer

Owns orchestration, command handlers, scheduler, initialization, and ports.

Important files:

- [application/ports.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/ports.py:1)
- [application/initialization_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/initialization_service.py:1)
- [application/execution_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/execution_service.py:1)

### Domain layer

Owns strategy rules and decision logic. No exchange API logic belongs here.

Important files:

- [domain/models.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/models.py:1)
- [domain/signals.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/signals.py:1)
- [domain/planner.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/planner.py:1)
- [domain/execution.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/execution.py:1)

### Infrastructure layer

Owns exchange access, data sync, DB persistence side effects, and notifications.

Important files:

- [infrastructure/bybit_spot.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/bybit_spot.py:1)
- [infrastructure/execution_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/execution_service.py:1)
- [infrastructure/notifications.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/notifications.py:1)

## Database Layout

The bot uses two PostgreSQL schemas:

- `_candles_trading_data`
- `_spot_trading_bot`

### Candle tables

For each tracked symbol, the runtime uses:

- `<symbol>_1d`
- `<symbol>_4h`

### Runtime tables

The bot maintains:

- `position_state`
  - current quantity
  - average entry price
  - total cost
  - `entry_count`
- `position_exit_state`
  - partial take-profit state
  - `first_take_profit_done`
  - partial-exit metadata
- `order_ledger`
  - execution history
  - dry-run / notification-only rows
  - no-loss audit fields
  - signal deduplication fields

Schema creation and migration entrypoints:

- [utils/db_actions.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/db_actions.py:1)
- [utils/create_tables.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/create_tables.py:1)
- [utils/run_migrations.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/run_migrations.py:1)
- [migrations](/home/yurii/Proj/pompilo/spot-greenwich-bot/migrations)

## CLI Commands

### Sync D1 candles

```bash
python3 main.py sync --period 365
```

### Sync 3 years of D1 candles

```bash
python3 main.py sync-3y
```

### Sync 4H candles

```bash
python3 main.py sync-4h --period 180
```

### Sync both D1 and 4H datasets

```bash
python3 main.py sync-full
```

### Run one analysis cycle

```bash
python3 main.py analyze --symbol ETHUSDT --dry-run
```

### Run one analysis cycle in notification-only mode

```bash
python3 main.py analyze --symbol ETHUSDT --notification-only
```

Notes:

- plain `analyze` is a one-off run without scheduling
- it initializes runtime tables before analysis
- current public CLI timeframe choice is `4h`
- `analyze --notification-only` is different: it switches to the live scheduler path for the selected symbols and keeps running like live mode, but without placing exchange orders

### Create runtime tables

```bash
python3 main.py init-db
```

### Run SQL migrations

```bash
python3 main.py migrate
```

### Run scheduled live mode

```bash
python3 main.py
```

### Run scheduled mode in dry-run

```bash
python3 main.py --dry-run
```

### Run scheduled mode in notification-only

```bash
python3 main.py --notification-only
```

## Scheduler Behavior

The live scheduler currently runs on `4H` UTC candle closes:

- `00:00:01`
- `04:00:01`
- `08:00:01`
- `12:00:01`
- `16:00:01`
- `20:00:01`

The CLI still exposes daily target settings in config, but the current live loop is `4H`-driven.

## Runtime Modes

### Normal mode

- real exchange orders are allowed
- runtime state is updated from real executions

### `--dry-run`

`dry-run` is a CLI execution mode.

- signals are calculated
- execution decisions are built
- exchange orders are not placed
- dry-run ledger rows are written
- notifications are still sent

### `NOTIFICATION_ONLY_MODE=true`

`notification-only` is an environment-controlled runtime mode.

- signals are calculated
- execution decisions are built
- exchange orders are not placed
- notification-only ledger rows are written
- notifications are still sent

### `--notification-only`

`notification-only` can also be enabled from the CLI for the current process only.

- `python3 main.py analyze --symbol ETHUSDT --notification-only`
- `python3 main.py --notification-only`

When the CLI flag is present, it overrides the environment default for that run.

Behavior differs by command:

- `python3 main.py --notification-only` runs the normal scheduled live loop without placing exchange orders
- `python3 main.py analyze --symbol ETHUSDT --notification-only` also uses the scheduler/live path for the selected symbols, rather than the usual one-off analyze path

`dry-run` takes precedence when both are enabled for the same command.

## Environment Configuration

The runtime configuration is defined in:

- [utils/config.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/config.py:1)
- [.env](/home/yurii/Proj/pompilo/spot-greenwich-bot/.env:1)

Environment loading order:

1. `.env.production`
2. `.env`

### Database

- `DB_HOST`
- `DB_PORT`
- `DATABASE`
- `DB_USER`
- `DB_PASSWORD`

### Exchange and API

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_REST_ENDPOINT`
- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `BYBIT_API_ENDPOINT`
- `BYBIT_RECV_WINDOW`

### App and notifications

- `APP_TIMEZONE`
- `APP_ENV`
- `LOG_LEVEL`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### Data sync and scheduler

- `H4_ANALYSIS_DAYS`
- `H4_INCREMENTAL_SYNC_DAYS`
- `DAILY_TARGET_HOUR`
- `DAILY_TARGET_MINUTE`
- `DAILY_TARGET_SECOND`
- `H4_SCHEDULER_ENABLED`

### Strategy parameters

- `GREENWICH_LENGTH`
- `GREENWICH_BASIS_TYPE`
- `GREENWICH_MULTIPLIER_1`
- `GREENWICH_MULTIPLIER_2`
- `GREENWICH_MULTIPLIER_3`
- `ANALYSIS_WINDOW`
- `H4_ANALYSIS_WINDOW`
- `D1_REGIME_FILTER_ENABLED`
- `ANTI_CRASH_BUY_BLOCK_ENABLED`
- `ANTI_CRASH_LOOKBACK_CANDLES`
- `ANTI_CRASH_MAX_DROP_RATIO`
- `CONFIRMATION_CANDLE_ENABLED`
- `ATR_POSITION_SIZING_ENABLED`
- `ATR_POSITION_SIZING_MEDIAN_WINDOW`
- `ATR_POSITION_SIZING_MIN_MULTIPLIER`
- `ATR_POSITION_SIZING_MAX_MULTIPLIER`

### Execution and risk controls

- `ORDER_DEPOSIT_PERCENT`
- `AVERAGING_ENTRY_LIMIT`
- `AVERAGING_ENTRY_2_SIZE_PERCENT`
- `AVERAGING_ENTRY_3_SIZE_PERCENT`
- `MIN_PROFIT_RATIO`
- `NO_LOSS_GUARD_ENABLED`
- `BUY_PRICE_GUARD_ENABLED`
- `BUY_PRICE_GUARD_MAX_DEVIATION_RATIO`
- `PORTFOLIO_CAP_ENABLED`
- `PORTFOLIO_POSITION_LIMIT`
- `PORTFOLIO_PRIORITY_SYMBOLS`
- `NOTIFICATION_ONLY_MODE`

## Tests

The repository contains a local unit-test suite under [tests](/home/yurii/Proj/pompilo/spot-greenwich-bot/tests).

Current state:

- full local suite: `99 passed`
- the suite does not require real exchange credentials
- the suite does not require a live PostgreSQL instance
- exchange, DB, and notification boundaries are mocked in unit tests

Run all tests:

```bash
pytest -q
```

## Related Documents

- [STRATEGY_IMPROVEMENTS.md](/home/yurii/Proj/pompilo/spot-greenwich-bot/STRATEGY_IMPROVEMENTS.md:1)
- [STANDARTS.md](/home/yurii/Proj/pompilo/spot-greenwich-bot/STANDARTS.md:1)
- [RELEASE_PLAN.md](/home/yurii/Proj/pompilo/spot-greenwich-bot/RELEASE_PLAN.md:1)

## Notes

- [index.html](/home/yurii/Proj/pompilo/spot-greenwich-bot/index.html:1) is not part of the Python runtime. It contains TradingView Pine Script material.
- [infrastructure/state_store.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/state_store.py:1) is currently not part of the active runtime flow.
