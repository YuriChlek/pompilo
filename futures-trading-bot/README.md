# Pompilo Futures Trading Bot

## Опис

Бот для торгівлі крипто-ф'ючерсами з одним активним торговим шляхом: `trend_breakout`.

Проєкт уміє:
- синхронізувати H1-свічки з Binance у PostgreSQL;
- будувати market context через індикаторний шар;
- генерувати сигнал `trend_breakout`;
- відкривати позиції через Bybit;
- супроводжувати відкриту позицію в live-режимі;
- надсилати сповіщення в Telegram;
- запускати database-backed backtest.

## Активна стратегія

У кодовій базі залишена одна активна стратегія: `trend_breakout`.

Логіка входу:
- торгується тільки підтверджений трендовий режим;
- вхід відбувається на пробій локального діапазону на закритті свічки;
- використовується `ATR`-based stop loss;
- розмір позиції рахується через risk-based sizing.

Логіка супроводу позиції:
- частковий вихід і переведення в breakeven після досягнення заданого `R`;
- далі стоп пересувається trailing-логікою;
- опційно позиція може бути закрита при deteriorated regime;
- нові входи проходять через portfolio-level risk controls:
  `max_open_positions`, `max_portfolio_heat`, cluster limits, `daily_loss_stop`.

Telegram:
- повідомлення про нову позицію відправляється тільки після успішного відкриття ордера;
- окреме повідомлення про breakeven відправляється тільки після фактичного stop-management update.

## Архітектура

- `main.py` — top-level CLI entrypoint
- `trading/application/` — trading cycle, bootstrap, scheduler
- `trading/domain/` — signal generation, regime logic, execution rules, portfolio admission
- `trading/infrastructure/` — Bybit, Binance sync, Telegram, market data adapters
- `indicators/` — H1/H4/D1 market context, range detection, ATR, volume, fractals
- `backtesting/` — replay, execution simulator, portfolio aggregation, reporting
- `api/` — sync historical candles from Binance
- `utils/` — config, DB helpers, table setup
- `tests/` — unit and contract tests

## Вимоги

- Python 3.11+
- PostgreSQL
- `.venv` з залежностями з `requirements.txt`

Основні бібліотеки:
- `asyncio`
- `asyncpg`, `sqlalchemy`
- `pandas`, `numpy`, `pandas-ta`
- `requests`, `httpx`, `tenacity`

## Встановлення

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Після цього потрібно:
- підняти PostgreSQL;
- створити БД `pompilo_db`;
- налаштувати `.env` у корені репозиторію.

## Мінімальний `.env`

```env
DB_HOST=localhost
DB_PORT=5432
DATABASE=pompilo_db
DB_USER=admin
DB_PASS=admin_pass

BYBIT_API_KEY=
BYBIT_API_SECRET=
BYBIT_API_ENDPOINT=https://api-demo.bybit.com
BYBIT_RECV_WINDOW=10000
LIMIT_ORDER_MAX_AGE=24
ANALYSIS_CANDLE_LIMIT=1500

TOKEN=
CHAT_ID=
APP_TIMEZONE=Europe/Kyiv
```

Додатково використовуються runtime flags для live stop management, зокрема:
- `BREAKEVEN_TRIGGER_R`
- `ENABLE_BREAKEVEN_STOP_MANAGEMENT`
- `ENABLE_BREAKEVEN_PARTIAL_CLOSE`

## Запуск

### Live mode

Без аргументів бот запускає live scheduler:

```bash
python3 main.py
```

Що робить live mode:
- створює потрібні таблиці;
- синхронізує market data;
- проходить по `TRADING_SYMBOLS`;
- супроводжує відкриті позиції;
- генерує нові `trend_breakout` сигнали;
- виконує ордери через Bybit;
- шле Telegram only-after-success.

### Sync mode

```bash
python3 main.py sync --period 365
```

### Backtest через `main.py`

```bash
python3 main.py backtest --symbol SOLUSDT --period 30
```

### Розширений backtest CLI

```bash
PYTHONPATH=. ./.venv/bin/python backtesting/run_backtest.py \
  --symbol SOLUSDT \
  --from 2026-03-07 \
  --to 2026-03-21 \
  --output-json artifacts/backtest_solusdt.json
```

Доступні ключові параметри:
- `--symbol` можна передавати кілька разів;
- `--lookback-candles`
- `--min-candles`
- `--indicator-history-period`
- `--no-reversal`
- `--intrabar-exit-priority`
- `--output-json`

## Що показує backtest

Backtest зараз повертає:
- базову trade statistics summary;
- `strategy_stats`;
- `signal_counts`, `filled_order_counts`, `skipped_signal_counts`;
- `exit_reason_counts`;
- `pnl_by_regime`;
- `expectancy_by_setup`;
- `max_drawdown_by_cluster`;
- повний serialized trade journal.

Детальніше див. [backtesting/README.md](/home/yurii/Proj/pompilo/futures-trading-bot/backtesting/README.md).

## Тести

Повний запуск:

```bash
./.venv/bin/python -m unittest discover -q
```

Корисні вибіркові модулі:

```bash
./.venv/bin/python -m unittest -q \
  tests.test_strategy_signal_logic \
  tests.test_execution_contracts \
  tests.test_backtesting_module
```

## Примітки

- Backtest і sync залежать від PostgreSQL.
- Live-режим додатково залежить від валідних Bybit і Telegram credentials.
- У репозиторії навмисно залишений один активний strategy path; старі strategy families більше не є частиною runtime.
