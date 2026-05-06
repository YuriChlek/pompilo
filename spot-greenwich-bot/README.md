# Spot Trading Bot

## Overview

`spot-trading-bot` is a Binance spot trading bot built around a Greenwich signal strategy and a layered `application / domain / infrastructure` architecture.

The bot:

- syncs daily candles from Binance into PostgreSQL
- reads candle history from PostgreSQL
- generates `BUY` and `SELL` signals from the Greenwich indicator
- reconciles local position state with Binance balances and trades
- executes spot market orders on Binance
- persists position state and order ledger records
- supports scheduled live runs and one-off analysis runs

## Trading Logic

The strategy is intentionally narrow:

- `BUY`: recovery above `lower3`
- `SELL`: break below `upper2`

Signal generation uses:

- [indicators/grinvich.py](/home/yurii/Proj/pompilo/spot-trading-bot/indicators/grinvich.py:1)
- [domain/signals.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/signals.py:1)
- [domain/execution.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/execution.py:1)
- [domain/planner.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/planner.py:1)

This project does not implement grid trading. It uses a single-signal spot execution model.

## Architecture

### Application

- [application/bootstrap.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/bootstrap.py:1)
- [application/ports.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/ports.py:1)
- [application/initialization_service.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/initialization_service.py:1)
- [application/trading_cycle_service.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/trading_cycle_service.py:1)
- [application/execution_service.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/execution_service.py:1)
- [application/scheduler.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/scheduler.py:1)
- [application/command_dispatcher.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/command_dispatcher.py:1)
- [application/runtime_commands.py](/home/yurii/Proj/pompilo/spot-trading-bot/application/runtime_commands.py:1)

### Domain

- [domain/models.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/models.py:1)
- [domain/signals.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/signals.py:1)
- [domain/execution.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/execution.py:1)
- [domain/planner.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/planner.py:1)

### Infrastructure

- [infrastructure/bybit_spot.py](/home/yurii/Proj/pompilo/spot-trading-bot/infrastructure/bybit_spot.py:1)
- [infrastructure/execution_service.py](/home/yurii/Proj/pompilo/spot-trading-bot/infrastructure/execution_service.py:1)
- [infrastructure/market_data_provider.py](/home/yurii/Proj/pompilo/spot-trading-bot/infrastructure/market_data_provider.py:1)
- [infrastructure/market_data_synchronizer.py](/home/yurii/Proj/pompilo/spot-trading-bot/infrastructure/market_data_synchronizer.py:1)
- [infrastructure/notifications.py](/home/yurii/Proj/pompilo/spot-trading-bot/infrastructure/notifications.py:1)

## CLI Commands

### Sync candles

```bash
python3 main.py sync --period 365
```

### Sync 3 years of candles

```bash
python3 main.py sync-3y
```

### Run one analysis cycle

```bash
python3 main.py analyze --symbol ETHUSDT --dry-run
```

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
python3 main.py --dry-run
```

## Environment

The bot uses settings from environment variables loaded by [utils/config.py](/home/yurii/Proj/pompilo/spot-trading-bot/utils/config.py:1).

Important settings include:

- PostgreSQL connection settings
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- daily scheduler target time
- Greenwich indicator parameters
- order quote amount
- profit threshold

## Tests

The repository contains lightweight unit tests under [tests](/home/yurii/Proj/pompilo/spot-trading-bot/tests/__init__.py:1).

In the current environment, the following test families are runnable without a fully provisioned dependency stack:

- command dispatcher
- runtime commands
- initialization service
- trading cycle service
- execution service
- scheduler
- execution policy
- dry-run flow
- reconciliation helpers
- Binance filters
- signal-generation fixtures

Some broader checks still require external packages such as `pandas` and `asyncpg`.

## Notes

- [index.html](/home/yurii/Proj/pompilo/spot-trading-bot/index.html:1) is not part of the production runtime. It contains TradingView Pine Script rather than browser-executed Python.
- [infrastructure/state_store.py](/home/yurii/Proj/pompilo/spot-trading-bot/infrastructure/state_store.py:1) is currently unused.
