# Order Flow Scalp Bot

Точка входу: [main.py](/home/yurii/my-proj/pompilo/_OF-rain-bot/main.py)

Проєкт уже працює через canonical runtime path:
- `main.py`
- `trading/application/runner.py`
- `trading/application/bootstrap.py`
- `trading/application/runtime.py`

Legacy namespace `orderflow/` видалений.

## Що робить бот

- отримує spot market data з `Bybit`, `Binance` та `OKX`
- будує signal по order flow та liquidity walls
- виконує ордери на `Bybit linear futures`
- зберігає runtime, order і position events у PostgreSQL
- надсилає Telegram notifications

## Режими запуску

### Live

```bash
python3 main.py
```

або

```bash
.venv/bin/python main.py
```

### Dry Run

```bash
python3 main.py --dry-run
```

або

```bash
.venv/bin/python main.py --dry-run
```

У `dry-run` бот:
- аналізує ринок як у live
- рахує `entry`, `stop loss`, `take profit`, `size`
- пише події в БД
- не відправляє реальні ордери

## CLI

Поточний entrypoint підтримує:
- запуск без аргументів
- `--dry-run`

Довідка:

```bash
python3 main.py --help
```

## Архітектура

Актуальна структура:
- `trading/application/*` — runtime orchestration, bootstrap, runner, ports
- `trading/domain/*` — signal generation, execution rules, risk logic, typed models
- `trading/infrastructure/*` — Bybit adapters, market data adapters, notifications, storage
- `utils/*` — config, env loading, DB pool
- `tests/*` — unit/regression tests

## Основні env-змінні

### Database

- `DB_HOST` — PostgreSQL host
- `DB_PORT` — PostgreSQL port
- `DB_USER` — PostgreSQL user
- `DB_PASSWORD` — PostgreSQL password
- `DATABASE` — PostgreSQL database name

### Bybit

- `BYBIT_API_KEY` — trading API key
- `BYBIT_API_SECRET` — trading API secret
- `BYBIT_TRADING_API_ENDPOINT` — REST endpoint для execution
- `BYBIT_MARKET_DATA_API_ENDPOINT` — REST endpoint для bootstrap market data
- `BYBIT_MARKET_DATA_WS_ENDPOINT` — public WS для spot market data
- `BYBIT_PRIVATE_WS_ENDPOINT` — private WS для order/execution/position events
- `BYBIT_RECV_WINDOW` — receive window для signed requests

### Trading / Risk

- `STOP_LOSS_SIZE` — фіксований stop loss у відсотках від ціни входу
- `ORDERFLOW_RISK_PER_TRADE_PCT` — ризик на одну угоду у відсотках від equity
- `ORDERFLOW_TAKE_PROFIT_R_MULTIPLE` — take profit у `R`
- `ORDERFLOW_MAX_BASIS_BPS` — максимально допустимий basis між spot і futures
- `ORDERFLOW_SYMBOLS` — список торгових символів

### Notifications

- `TOKEN` — Telegram bot token
- `CHAT_ID` — Telegram chat/channel id
- `APP_TIMEZONE` — timezone для повідомлень і логів

Повний приклад дивись у [ .env ](/home/yurii/my-proj/pompilo/_OF-rain-bot/.env:1).

## Поточна логіка stop loss

Для нових позицій stop loss рахується як фіксований відступ від entry price:
- `long`: `entry * (1 - STOP_LOSS_SIZE / 100)`
- `short`: `entry * (1 + STOP_LOSS_SIZE / 100)`

За замовчуванням `STOP_LOSS_SIZE=0.5`, тобто `0.5%`.

## Тести

Запуск основного regression-набору:

```bash
.venv/bin/python -m unittest
```

Актуальний локальний набір unit tests проходить успішно.
