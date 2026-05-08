# Spot Greenwich Bot

## Огляд

`spot-greenwich-bot` це бот для spot-торгівлі з чистим поділом на шари `application / domain / infrastructure`.

Поточний runtime робить таке:

- синхронізує `D1` і `4H` свічки з Binance у PostgreSQL
- використовує `D1` як фільтр ринкового режиму
- використовує `4H` як робочий таймфрейм для входів і виходів
- генерує Greenwich-сигнали `BUY` і `SELL`
- звіряє локальний runtime-state з балансами і трейдами Bybit
- виконує spot market orders на Bybit
- зберігає runtime-state і історію виконання в PostgreSQL
- відправляє логові та Telegram-сповіщення
- підтримує режими `dry-run` і `notification-only`

Це не grid bot і не система виставлення пачки відкладених ордерів. Це signal-driven bot для spot execution.

## Коротко про стратегію

Базова стратегія — Greenwich mean reversion:

- `BUY`: повернення назад вище `lower3`
- `SELL`: crossunder нижче `upper2`

Поверх базового Greenwich-сигналу поточний runtime додає:

- `D1` regime filter: `death cross` + `ADX >= 30` блокує нові `BUY` на `4H`
- confirmation candle для `BUY`
- anti-crash блокування після різкого падіння
- ліміт усереднень зі зменшенням розміру наступних входів
- частковий take-profit на `upper1`
- ATR-нормалізований розмір `BUY`
- `BUY` market price guard
- portfolio cap з пріоритетними символами
- no-loss guard для `SELL`

Основна реалізація цієї логіки знаходиться в:

- [domain/signals.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/signals.py:1)
- [domain/planner.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/planner.py:1)
- [domain/execution.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/execution.py:1)
- [infrastructure/execution_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/execution_service.py:1)

## Правила стратегії

### Коли бот входить у позицію

Бот відкриває нову позицію або докуповує існуючу тільки тоді, коли одночасно виконуються всі умови:

1. На `4H` є базовий Greenwich `BUY` setup.
   - Суть сигналу: відновлення після перепроданого стану біля `lower3`.
2. `D1` regime filter дозволяє новий `BUY`.
   - Якщо на `D1` виконується `WMA(50) < WMA(200)` і `ADX >= 30`, нові `BUY` блокуються.
3. Логіка confirmation candle дозволяє вхід.
   - Свіжий `BUY` не виконується миттєво.
   - Наступна закрита свічка повинна підтвердити відновлення закриттям вище `lower3`.
4. Проходить anti-crash filter.
   - Якщо недавнє падіння за lookback занадто велике, `BUY` блокується.
5. Execution policy дозволяє вхід.
   - Для вже відкритої позиції новий `BUY` має бути за кращою ціною, ніж поточна середня ціна входу.
   - Ліміт усереднень не повинен бути перевищений.
   - Portfolio cap повинен дозволяти відкриття нового символу, якщо це нова позиція.
6. Infrastructure price guard дозволяє market order.
   - Якщо поточна ринкова ціна занадто високо відійшла від сигнальної, `BUY` блокується.

### Як працює усереднення

Бот може докуповувати позицію, але лише в контрольованому режимі:

- перший вхід: `100%` базового розміру
- другий вхід: `60%` базового розміру
- третій вхід: `30%` базового розміру
- четвертий і далі: блокуються

Тобто бот не усереднює позицію безкінечно під час затяжного падіння.

### Коли бот виходить із позиції

У бота є два основні шляхи виходу:

1. Частковий take-profit:
   - якщо позиція відкрита і ціна досягає `upper1`, бот продає `50%`
   - цей partial exit може статись лише один раз на позицію
2. Фінальний вихід:
   - коли з’являється звичайний Greenwich `SELL`, бот продає залишок позиції
   - базова умова `SELL`: crossunder нижче `upper2`

### Коли бот не торгує

Бот свідомо пропускає угоду в таких випадках:

- немає валідного Greenwich signal
- `D1` regime блокує новий `BUY`
- confirmation candle ще не з’явилась
- confirmation candle не пройшла підтвердження
- anti-crash filter блокує вхід
- новий `BUY` не кращий за поточну середню ціну входу
- ліміт усереднень уже вичерпано
- portfolio cap блокує відкриття ще одного символу
- ринкова ціна занадто далеко пішла вгору від `BUY` signal price
- нормалізована кількість ордера нижча за мінімальні фільтри Bybit
- `SELL` не дає достатнього прибутку відносно `MIN_PROFIT_RATIO`
- no-loss guard бачить, що поточна жива ціна продажу вже впала нижче мінімального прибуткового рівня
- такий самий сигнал уже був виконаний і вважається duplicate

Тобто бот від початку зроблений вибірковим: він краще пропустить угоду, ніж увійде в слабкий або погано виконуваний setup.

## Як проходить один торговий цикл

Один цикл роботи виглядає так:

1. Завантажується історія свічок із PostgreSQL.
2. Завантажується і звіряється поточний стан позиції з Bybit та локальною БД.
3. Будується торговий план:
   - raw Greenwich signal
   - signal filters
   - execution decision
4. У multi-symbol циклі застосовуються portfolio-level правила допуску.
5. Рішення виконується через Bybit executor або симулюється в `dry-run` / `notification-only`.
6. Оновлюються runtime-state і `order_ledger`.
7. Відправляються сповіщення.

Основні entrypoints і orchestration:

- [main.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/main.py:1)
- [application/bootstrap.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/bootstrap.py:1)
- [application/runtime_commands.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/runtime_commands.py:1)
- [application/trading_cycle_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/trading_cycle_service.py:1)
- [application/scheduler.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/scheduler.py:1)

## Архітектура

### Шар Application

Відповідає за orchestration, command handlers, scheduler, initialization та порти.

Ключові файли:

- [application/ports.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/ports.py:1)
- [application/initialization_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/initialization_service.py:1)
- [application/execution_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/application/execution_service.py:1)

### Шар Domain

Містить правила стратегії та логіку прийняття рішень. Біржової логіки тут бути не повинно.

Ключові файли:

- [domain/models.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/models.py:1)
- [domain/signals.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/signals.py:1)
- [domain/planner.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/planner.py:1)
- [domain/execution.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/domain/execution.py:1)

### Шар Infrastructure

Містить доступ до біржі, синхронізацію даних, side effects із БД і нотифікації.

Ключові файли:

- [infrastructure/bybit_spot.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/bybit_spot.py:1)
- [infrastructure/execution_service.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/execution_service.py:1)
- [infrastructure/notifications.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/notifications.py:1)

## Структура БД

Бот використовує дві PostgreSQL schema:

- `_candles_trading_data`
- `_spot_trading_bot`

### Candle tables

Для кожного символу використовуються таблиці:

- `<symbol>_1d`
- `<symbol>_4h`

### Runtime tables

У runtime підтримуються:

- `position_state`
  - поточна кількість
  - середня ціна входу
  - total cost
  - `entry_count`
- `position_exit_state`
  - state часткового виходу
  - `first_take_profit_done`
  - метадані partial exit
- `order_ledger`
  - історія виконань
  - записи `dry-run` / `notification-only`
  - audit-поля для no-loss
  - поля для signal deduplication

Файли, пов’язані зі schema creation та migrations:

- [utils/db_actions.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/db_actions.py:1)
- [utils/create_tables.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/create_tables.py:1)
- [utils/run_migrations.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/run_migrations.py:1)
- [migrations](/home/yurii/Proj/pompilo/spot-greenwich-bot/migrations)

## CLI-команди

### Синхронізація D1-свічок

```bash
python3 main.py sync --period 365
```

### Синхронізація 3 років D1-історії

```bash
python3 main.py sync-3y
```

### Синхронізація 4H-свічок

```bash
python3 main.py sync-4h --period 180
```

### Повна синхронізація D1 і 4H

```bash
python3 main.py sync-full
```

### Один цикл аналізу

```bash
python3 main.py analyze --symbol ETHUSDT --dry-run
```

### Один цикл аналізу в notification-only режимі

```bash
python3 main.py analyze --symbol ETHUSDT --notification-only
```

Примітки:

- звичайний `analyze` це разовий запуск без scheduler-а
- перед аналізом ініціалізуються runtime tables
- публічний CLI зараз працює з `4h` як робочим timeframe
- `analyze --notification-only` працює інакше: він переходить у scheduler/live path для вибраних символів і продовжує працювати як live mode, але без виставлення біржових ордерів

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
python3 main.py
```

### Плановий режим у dry-run

```bash
python3 main.py --dry-run
```

### Плановий режим у notification-only

```bash
python3 main.py --notification-only
```

## Поведінка scheduler-а

Поточний live scheduler працює по UTC-закриттях `4H` свічок:

- `00:00:01`
- `04:00:01`
- `08:00:01`
- `12:00:01`
- `16:00:01`
- `20:00:01`

У конфігурації ще є daily target settings, але фактичний live loop зараз `4H`-орієнтований.

## Режими роботи

### Звичайний режим

- реальні біржові ордери дозволені
- runtime-state оновлюється з реальних виконань

### `--dry-run`

`dry-run` це CLI execution mode.

- сигнали рахуються
- execution decisions будуються
- біржові ордери не виставляються
- у ledger пишуться dry-run записи
- сповіщення все одно відправляються

### `NOTIFICATION_ONLY_MODE=true`

`notification-only` це runtime mode, який вмикається через env.

- сигнали рахуються
- execution decisions будуються
- біржові ордери не виставляються
- у ledger пишуться notification-only записи
- сповіщення все одно відправляються

### `--notification-only`

`notification-only` також можна ввімкнути через CLI тільки для поточного запуску.

- `python3 main.py analyze --symbol ETHUSDT --notification-only`
- `python3 main.py --notification-only`

Якщо CLI-прапорець передано, він має пріоритет над env-значенням для цього конкретного запуску.

Поведінка залежить від команди:

- `python3 main.py --notification-only` запускає звичайний scheduled live loop без виставлення ордерів
- `python3 main.py analyze --symbol ETHUSDT --notification-only` теж переходить у scheduler/live path для вибраних символів, а не лишається звичайним one-shot analyze

Якщо одночасно ввімкнено і `dry-run`, і `notification-only`, пріоритет має `dry-run`.

## Налаштування через environment

Runtime-конфігурація описана в:

- [utils/config.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/utils/config.py:1)
- [.env](/home/yurii/Proj/pompilo/spot-greenwich-bot/.env:1)

Порядок завантаження env:

1. `.env.production`
2. `.env`

### База даних

- `DB_HOST`
- `DB_PORT`
- `DATABASE`
- `DB_USER`
- `DB_PASSWORD`

### Біржа і API

- `BINANCE_API_KEY`
- `BINANCE_API_SECRET`
- `BINANCE_REST_ENDPOINT`
- `BYBIT_API_KEY`
- `BYBIT_API_SECRET`
- `BYBIT_API_ENDPOINT`
- `BYBIT_RECV_WINDOW`

### Застосунок і нотифікації

- `APP_TIMEZONE`
- `APP_ENV`
- `LOG_LEVEL`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### Синхронізація даних і scheduler

- `H4_ANALYSIS_DAYS`
- `H4_INCREMENTAL_SYNC_DAYS`
- `DAILY_TARGET_HOUR`
- `DAILY_TARGET_MINUTE`
- `DAILY_TARGET_SECOND`
- `H4_SCHEDULER_ENABLED`

### Параметри стратегії

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

### Виконання і контроль ризику

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

## Тести

У репозиторії є локальний unit-test suite у [tests](/home/yurii/Proj/pompilo/spot-greenwich-bot/tests).

Поточний стан:

- повний локальний suite: `99 passed`
- для прогону не потрібні реальні біржові credentials
- не потрібен live PostgreSQL
- межі біржі, БД і нотифікацій замокані на unit-рівні

Запуск усіх тестів:

```bash
pytest -q
```

## Пов’язані документи

- [STRATEGY_IMPROVEMENTS.md](/home/yurii/Proj/pompilo/spot-greenwich-bot/STRATEGY_IMPROVEMENTS.md:1)
- [STANDARTS.md](/home/yurii/Proj/pompilo/spot-greenwich-bot/STANDARTS.md:1)
- [RELEASE_PLAN.md](/home/yurii/Proj/pompilo/spot-greenwich-bot/RELEASE_PLAN.md:1)

## Примітки

- [index.html](/home/yurii/Proj/pompilo/spot-greenwich-bot/index.html:1) не є частиною Python runtime. Усередині нього знаходяться матеріали для TradingView Pine Script.
- [infrastructure/state_store.py](/home/yurii/Proj/pompilo/spot-greenwich-bot/infrastructure/state_store.py:1) зараз не входить до активного runtime flow.
