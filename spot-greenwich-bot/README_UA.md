# Spot Trading Bot

## Огляд

`spot-trading-bot` це бот для spot-торгівлі на Binance, побудований навколо Greenwich-стратегії та шарів `application / domain / infrastructure`.

Бот:

- синхронізує денні свічки з Binance у PostgreSQL
- читає історію свічок з PostgreSQL
- формує сигнали `BUY` і `SELL` за Greenwich-індикатором
- звіряє локальний стан позиції з балансами та трейдами Binance
- виконує spot market orders на Binance
- зберігає `position_state` і `order_ledger`
- підтримує плановий live-запуск і разовий режим аналізу

## Торгова логіка

Стратегія тут навмисно вузька:

- `BUY`: повернення вище `lower3`
- `SELL`: пробій нижче `upper2`

Джерела логіки:

- [indicators/grinvich.py](/home/yurii/Proj/pompilo/spot-trading-bot/indicators/grinvich.py:1)
- [domain/signals.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/signals.py:1)
- [domain/execution.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/execution.py:1)
- [domain/planner.py](/home/yurii/Proj/pompilo/spot-trading-bot/domain/planner.py:1)

Це не grid bot. Тут використовується проста signal-driven spot execution модель.

## Архітектура

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

## CLI-команди

### Синхронізація свічок

```bash
python3 main.py sync --period 365
```

### Синхронізація 3 років історії

```bash
python3 main.py sync-3y
```

### Один цикл аналізу

```bash
python3 main.py analyze --symbol ETHUSDT --dry-run
```

### Створення runtime-таблиць

```bash
python3 main.py init-db
```

### Запуск SQL-міграцій

```bash
python3 main.py migrate
```

### Плановий live-режим

```bash
python3 main.py --dry-run
```

## Налаштування середовища

Бот використовує змінні середовища, які читаються через [utils/config.py](/home/yurii/Proj/pompilo/spot-trading-bot/utils/config.py:1).

Ключові параметри:

- налаштування PostgreSQL
- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- час щоденного запуску scheduler-а
- параметри Greenwich-індикатора
- розмір ордера в quote-валюті
- мінімальний поріг прибутковості для продажу

## Тести

У репозиторії є unit-тести в [tests](/home/yurii/Proj/pompilo/spot-trading-bot/tests/__init__.py:1).

У поточному середовищі без повного dependency stack реально виконуються тести для:

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

Ширші перевірки все ще залежать від пакетів на кшталт `pandas` і `asyncpg`.

## Примітки

- [index.html](/home/yurii/Proj/pompilo/spot-trading-bot/index.html:1) не є частиною production runtime. Усередині нього знаходиться Pine Script для TradingView, а не Python-код для браузера.
- [infrastructure/state_store.py](/home/yurii/Proj/pompilo/spot-trading-bot/infrastructure/state_store.py:1) зараз не використовується.
