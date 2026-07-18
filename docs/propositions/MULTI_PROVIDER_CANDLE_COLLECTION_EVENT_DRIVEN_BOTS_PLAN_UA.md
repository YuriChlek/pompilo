# План переходу до multi-provider candle collection та event-driven запуску ботів

Документ описує цільову модель роботи `market-data-service` і `bot-platform-service`, у якій основний workflow зводиться до двох зрозумілих команд:

```bash
python -m market_data_service.main collect
python -m market_data_service.main candles:get --period 100d [--symbols BTCUSDT] [--timeframes 1h,4h,1d] [--provider auto]
```

Перша команда запускає збір свічок з доступного provider-а і записує їх у БД. Під час першого bootstrap/backfill history sync події не створюються і боти не запускаються. Після завершеного bootstrap команда створює події тільки для live/incremental ranges; `bot-platform-service` підхоплює ці події і запускає enabled ботів для відповідного symbol/timeframe.

Друга команда отримує свічки з біржі за вказаний lookback period від поточного UTC часу назад і записує отримані candles у БД як ручне завантаження тестових/історичних даних. Наприклад, `--period 1d`, `--period 100d`, `--period 1y`. За замовчуванням `--provider auto` використовує provider resolver: якщо symbol є на Binance, дані беруться з Binance; якщо symbol відсутній на Binance, але доступний на Bybit, дані беруться з Bybit. Команда потрібна для ручного завантаження тестових даних, перевірки provider-а, діагностики і точкового поповнення candles. `candles:get` не створює market-data events, не публікує Redis events і не запускає торгових ботів.

Якщо для `candles:get` не передати символи або таймфрейми явно, команда використовує symbols/timeframes з конфігурації сервісу.

Цільові таймфрейми збору:

```text
1h
4h
1d
```

Canonical timeframe-и в БД, config, CLI і bot-platform contracts мають залишатися у форматі `1h`, `4h`, `1d`. Це збігається з поточною схемою partitions і не потребує міграції існуючих candles. Provider adapter-и мають мапити ці canonical timeframe-и на біржові interval-и. Наприклад: Binance `1h -> 1h`, `4h -> 4h`, `1d -> 1d`; Bybit `1h -> 60`, `4h -> 240`, `1d -> D`.

Важливий принцип: непотрібний код і команди не позначаються як legacy, а видаляються після появи нового production flow і міграції тестів/документації.

## 1. Цільовий стан

### 1.1. Public CLI

У `market-data-service` залишаються дві основні user-facing команди:

```bash
python -m market_data_service.main collect
python -m market_data_service.main candles:get
```

Команда `collect` має підтримувати два режими:

```bash
python -m market_data_service.main collect --once
python -m market_data_service.main collect
```

`collect --once`:

- виконує один повний цикл збору;
- підходить для smoke, CI і ручної перевірки;
- завершується з exit code `0`, якщо всі configured/resolved source/symbol/timeframe ranges оброблені або контрольовано пропущені.

`collect` без `--once`:

- запускає `market-data-service` як long-running service;
- на кожній ітерації синхронізує closed candles для всіх symbols, заданих у `MARKET_DATA_PROVIDER_SYMBOLS`, і таймфреймів `1h`, `4h`, `1d` з `MARKET_DATA_TIMEFRAMES`;
- для кожного symbol визначає активний provider через provider resolver і priority list;
- пише candles/snapshots/batches у БД;
- не створює events під час першого bootstrap/backfill history sync;
- створює events після успішного commit тільки для live/incremental ranges після завершеного bootstrap;
- публікує live/incremental events у Redis Stream;
- має health/readiness/metrics;
- коректно завершується по `SIGTERM` / `SIGINT`.

Команда `candles:get`:

- звертається до provider-а, визначеного через `--provider`;
- у режимі `--provider auto` використовує provider resolver і priority list;
- отримує candles за обовʼязковий lookback period `--period`, який рахується назад від поточного UTC часу;
- записує отримані candles у БД через idempotent upsert;
- приймає опційні `--symbols` і `--timeframes`;
- якщо `--symbols` не передано, використовує `MARKET_DATA_PROVIDER_SYMBOLS`;
- якщо `--timeframes` не передано, використовує `MARKET_DATA_TIMEFRAMES`;
- підтримує batch fetch для всіх configured symbol/timeframe pairs;
- виводить у stdout JSON summary із range, provider/source, кількістю fetched/inserted/skipped candles і помилками;
- не створює market-data outbox events;
- не публікує Redis events;
- не запускає торгових ботів.

`candles:get` є manual test-data/backfill load path. Він має бути безпечним для повторного запуску: duplicate candles не мають створювати дублікати і не мають запускати event-driven bot flow.

### 1.2. Internal commands

Поточні технічні команди:

```bash
scheduler:run-once
sync:run-next
outbox:publish-once
symbols:sync
backfill
gaps:scan
outbox:replay
scheduler
outbox-publisher
db:revision
```

не мають залишатися основним workflow.

Після впровадження нового flow треба:

- перенести потрібну логіку у application services, які використовує `collect`;
- оновити smoke/tests на `collect --once`;
- видалити CLI routing для непотрібних команд;
- видалити README-інструкції для непотрібних команд;
- видалити Docker jobs для непотрібних команд, якщо вони більше не використовуються;
- видалити тести, які перевіряють тільки старі CLI-команди, або переписати їх на новий behavior.

Не потрібно залишати dead code з коментарями “legacy”.

## 2. Цільова архітектура

### 2.1. Market Data collect flow

`collect` має виконувати такий pipeline:

1. Завантажити settings.
2. Створити runtime container.
3. Переконатися, що migrations застосовані.
4. Завантажити provider priority list.
5. Для кожного configured symbol визначити active provider через provider resolver.
6. Не робити запити до provider-а, якщо для цього symbol/provider вже є актуальний запис `UNSUPPORTED` до `next_check_at`.
7. Для кожного resolved `source/symbol/timeframe` визначити closed candle range, який треба зібрати.
8. Забрати candles з active provider-а.
9. Нормалізувати і провалідувати candles.
10. Записати candles у `_market_data.market_candles` і повʼязані batch/snapshot records.
11. Якщо це live/incremental range після завершеного bootstrap для цієї `source/symbol/timeframe` пари, створити outbox event record в одній DB transaction з candles/batch/snapshot.
12. Якщо це перша bootstrap-синхронізація історії або manual backfill, не створювати outbox event і не запускати ботів.
13. Після успішного DB commit опублікувати outbox event у Redis Stream тільки якщо event був створений.
14. Якщо Redis publish не вдався, залишити outbox event у pending/retry стані без втрати записаних candles.
15. Зафіксувати metrics/logs.

Outbox event record для live/incremental range має створюватися в тій самій транзакції, що і candles/batch/snapshot. Redis-публікація не має виконуватися до commit цієї транзакції. Це прибирає ризик стану “candles записані, але event втрачений через crash між commit і створенням event”.

Bootstrap/backfill rule:

- перша синхронізація історії за 2 роки є bootstrap/backfill mode;
- bootstrap/backfill пише candles і batch/snapshot metadata у БД;
- bootstrap/backfill не створює market-data events;
- bootstrap/backfill не публікує Redis events;
- bootstrap/backfill не запускає bot-platform bot runs;
- bootstrap/backfill можна виконувати поступово кількома ітераціями, навіть якщо повна синхронізація історії займе значний час;
- bootstrap/backfill має поважати provider rate limits через pagination, concurrency limit, cooldown/backoff і максимальну кількість bootstrap chunks за одну ітерацію;
- після успішного завершення bootstrap для конкретної `source/symbol/timeframe` пари наступні incremental/live closed ranges можуть створювати events і запускати ботів.

Правило визначення range:

- якщо для `source/symbol/timeframe` у БД ще немає жодної успішно збереженої candle, це перший запуск для цієї пари;
- при першому запуску сервіс синхронізує candles за останні 2 роки до останньої безпечної closed candle на active provider;
- 2-річний bootstrap range не обовʼязково має бути синхронізований за одну ітерацію: сервіс може зберігати progress і продовжувати з останнього успішного bootstrap chunk;
- якщо active provider має історію для конкретного symbol/timeframe меншу за 2 роки, сервіс записує весь доступний provider range і не вважає відсутню старішу історію помилкою або gap-ом;
- якщо candles уже є, сервіс синхронізує діапазон після останньої успішно збереженої closed candle до останньої безпечної closed candle на active provider;
- якщо для resilience потрібен inclusive refetch останньої збереженої candle, цей refetch не має входити в event range і має бути поглинутий idempotent upsert;
- якщо сервіс був вимкнений або втрачав зʼєднання, наступна ітерація має добрати весь пропущений range, а не тільки останню candle;
- повторний запис уже наявних candles має бути idempotent через unique constraints/upsert behavior.

### 2.2. Multi-provider symbol resolver

Сервіс має підтримувати декілька джерел candles. Початковий priority list:

```bash
MARKET_DATA_PROVIDER_PRIORITY=binance,bybit
```

Правила resolution:

- для кожного configured symbol сервіс спочатку перевіряє Binance;
- якщо symbol доступний на Binance, active provider для цього symbol - `BINANCE_SPOT`;
- якщо symbol відсутній на Binance, сервіс перевіряє Bybit;
- якщо symbol доступний на Bybit, active provider для цього symbol - `BYBIT_SPOT`;
- якщо symbol відсутній на всіх provider-ах, він отримує status `UNSUPPORTED`;
- unsupported status кешується, щоб не робити зайві запити на біржу, де цієї монети немає;
- після TTL symbol можна перевірити повторно, бо монета могла зʼявитися на біржі пізніше.
- `UNSUPPORTED` кешується для конкретної пари `source/requested_symbol`;
- `UNSUPPORTED` на Binance не блокує перевірку Bybit;
- `TEMPORARY_ERROR` не можна трактувати як `UNSUPPORTED`; для нього потрібен коротший retry/backoff, а не довгий unsupported TTL.

Приклад:

```text
HYPEUSDT відсутній на Binance
HYPEUSDT доступний на Bybit
=> collect бере candles HYPEUSDT з BYBIT_SPOT
=> Binance не опитується повторно до next_check_at
```

Рекомендована таблиця або read model:

```text
_market_data.provider_symbol_availability
```

Поля:

```text
source
requested_symbol
provider_symbol
status              # SUPPORTED | UNSUPPORTED | TEMPORARY_ERROR
first_seen_at
last_checked_at
next_check_at
failure_reason
metadata
```

Env config:

```bash
MARKET_DATA_PROVIDER_PRIORITY=binance,bybit
MARKET_DATA_PROVIDER_AVAILABILITY_TTL_HOURS=24
MARKET_DATA_UNSUPPORTED_SYMBOL_RECHECK_HOURS=24
```

Термінологія provider/source:

- CLI/config provider alias: `binance`, `bybit`;
- persisted/event source: `BINANCE_SPOT`, `BYBIT_SPOT`;
- `MARKET_DATA_PROVIDER_PRIORITY` містить provider aliases, а resolver мапить їх на source enum;
- availability cache key використовує persisted source і requested symbol: `source/requested_symbol`.

Resolution result має бути persisted. `collect` не має кожну ітерацію робити network request до Binance для symbol, про який уже відомо, що він `UNSUPPORTED` на Binance до `next_check_at`.

### 2.3. Bot Platform event-driven flow

`bot-platform-service` має мати event consumer, який:

1. Читає Redis Stream market-data events.
2. Фільтрує тільки events типу `market_data.candles_collected`.
3. Перевіряє event schema/version.
4. Знаходить enabled bot instances, які підписані на `symbol/timeframe`.
5. Для кожного instance створює idempotent bot run.
6. Читає snapshot за `snapshot_id` з event через formal market-data contract.
7. Запускає platform-native adapter.
8. Пише `bot_runs`, `bot_run_events`, `bot_signals`, audit events.
9. Публікує downstream signal events, якщо це увімкнено.
10. ACK-ає Redis Stream delivery тільки після успішної обробки або контрольованого terminal failure.

Одна market-data подія не повинна створювати неконтрольовані дублікати bot runs.

### 2.4. Event contract

Подія після успішного збору candles:

```json
{
  "event_type": "market_data.candles_collected",
  "contract_version": "market-data-event.v1",
  "source": "BINANCE_SPOT",
  "symbol": "BTCUSDT",
  "provider_symbol": "BTCUSDT",
  "timeframe": "1h",
  "from": "2026-07-16T00:00:00Z",
  "to": "2026-07-16T01:00:00Z",
  "batch_id": "batch_id",
  "snapshot_id": "snapshot_id",
  "closed_at": "2026-07-16T01:00:00Z",
  "idempotency_key": "BINANCE_SPOT:BTCUSDT:1h:2026-07-16T01:00:00Z"
}
```

Обовʼязкові поля:

- `event_type`;
- `contract_version`;
- `source`;
- `symbol`;
- `provider_symbol`;
- `timeframe`;
- `from`;
- `to`;
- `batch_id`;
- `snapshot_id`;
- `closed_at`;
- `idempotency_key`.

`idempotency_key` має бути стабільним для одного source/symbol/timeframe/closed window.

Range contract:

- `from` включно;
- `to` виключно;
- тобто range має формат `[from, to)`;
- `closed_at` відповідає закриттю останньої fully closed candle у цьому event.

### 2.5. Retention і cleanup

Event-driven flow не має накопичувати службові події без обмеження.

Retention rules:

- Redis Stream `market-data-events` має 2-денне recovery window для delivery events;
- Redis Stream має використовувати trimming через time-based cleanup або bounded `MAXLEN`, розрахований під 2-денне recovery window з documented throughput assumption і safety margin;
- market-data outbox records у БД з terminal status `PUBLISHED` зберігаються не довше 5 днів;
- bot-platform processed/idempotency records для market-data events зберігаються не довше 5 днів;
- службові raw event/audit delivery records, які потрібні тільки для replay/debug market-data events, зберігаються не довше 5 днів;
- `PENDING` і retryable `FAILED` records не видаляються автоматично як звичайні published events, доки не будуть оброблені окремою operational policy;
- історичні candles у `_market_data.market_candles` не підпадають під 5-денний cleanup, бо вони потрібні для 2-річного bootstrap, gap recovery, bot runs і подальшого аналізу.

Рекомендовані env vars:

```bash
MARKET_DATA_REDIS_EVENT_RETENTION_DAYS=2
MARKET_DATA_OUTBOX_RETENTION_DAYS=5
BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS=5
BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS=5
```

Cleanup має бути idempotent, batch-based і безпечний для повторного запуску. Видалення старих записів не повинно ламати replay у межах активного recovery window.

### 2.6. DB indexes для читання candles і latest snapshots

Основні read paths мають бути покриті індексами до запуску event-driven bot flow:

- candles range lookup: `source + canonical_symbol + timeframe + open_time range`;
- event snapshot lookup: `snapshot_id`;
- latest complete snapshot lookup для readiness/manual diagnostics: `source + canonical_symbol + timeframe + completeness_status`, sort by `last_closed_candle_time DESC`, `snapshot_version DESC`, `created_at DESC`;
- snapshot membership read: `snapshot_id -> ordered candles`.

Для `_market_data.market_candles` потрібен unique/index:

```sql
UNIQUE (source, canonical_symbol, timeframe, open_time)
```

Він одночасно забезпечує idempotent insert і швидкий range lookup за symbol/timeframe.

Для `_market_data.market_snapshots` потрібен спеціальний індекс під latest complete snapshot:

```sql
CREATE INDEX market_snapshots_latest_complete_idx
ON _market_data.market_snapshots (
  source,
  canonical_symbol,
  timeframe,
  completeness_status,
  last_closed_candle_time DESC,
  snapshot_version DESC,
  created_at DESC
);
```

Після реалізації треба перевірити `EXPLAIN ANALYZE` для:

- отримання candles за period для одного `source/symbol/timeframe`;
- snapshot lookup by event `snapshot_id`;
- `get_latest_complete_snapshot` для readiness/manual diagnostics;
- читання candles за `snapshot_id`.

## 3. Що видаляємо або замінюємо

### 3.1. Market Data CLI

Після реалізації `collect` і `candles:get` потрібно видалити з public CLI:

- `scheduler:run-once`;
- `sync:run-next`;
- `outbox:publish-once`;
- `symbols:sync`;
- `backfill`;
- `gaps:scan`;
- `outbox:replay`;
- `scheduler`;
- `outbox-publisher`;
- `db:revision`.

`healthcheck` можна залишити, якщо Docker healthcheck використовує CLI wrapper для HTTP readiness. Це не user-facing бізнес-команда, а operational command.

`db:check` можна залишити тільки якщо воно реально потрібне для ops; якщо ні, видалити разом з іншими допоміжними CLI-командами.

### 3.2. Docker jobs

Залишити:

- `market_data_migrate`;
- long-running `market_data` або Docker command для `collect`;
- `bot_platform_migrate`;
- `bot_modules_sync`, якщо metadata sync залишається окремим release/deploy step.

Видалити або замінити:

- `market_data_symbols_sync`;
- `market_data_backfill`;
- будь-які Docker jobs, які існують тільки для старого розбитого ingest flow.

Якщо symbol sync потрібен `collect`, він має бути внутрішнім кроком collect pipeline, а не окремим ручним job.

### 3.3. README та docs

Після зміни workflow README має описувати тільки:

- як встановити залежності;
- як підняти PostgreSQL/Redis;
- як застосувати migrations;
- як запустити `collect`;
- як запустити `candles:get`;
- як налаштувати provider priority Binance -> Bybit;
- як запустити Docker flow;
- як перевірити event-driven запуск ботів.

Старі команди не мають залишатися в документації як “legacy”.

## 4. Етапи реалізації

Етапи навмисно малі. Кожен етап має давати перевірний результат і не вимагати одночасної зміни всіх сервісів.

### Етап 1. Зафіксувати CLI contract для `collect`

Мета: додати public CLI entrypoint для запуску збору candles без перенесення ingest logic.

Завдання:

- додати parser/route для `collect`;
- додати опцію `--once`;
- старі команди поки не видаляти, щоб не ламати тести до переносу logic;
- додати unit tests для `collect --help` і validation.

Acceptance criteria:

- `python -m market_data_service.main collect --help` працює;
- invalid args повертають non-zero exit code.

### Етап 2. Зафіксувати CLI contract для `candles:get`

Мета: додати public CLI entrypoint для ручного отримання candles за lookback period.

Завдання:

- додати parser/route для `candles:get`;
- додати обовʼязковий аргумент `--period`;
- підтримати period units `h`, `d`, `w`, `mo`, `y`, наприклад `12h`, `1d`, `100d`, `2w`, `3mo`, `1y`;
- обчислювати range як `[now - period, now)`, де `now` - поточний UTC час запуску команди;
- додати опційні аргументи `--symbols`, `--timeframes`, `--provider`;
- якщо `--symbols` або `--timeframes` не передані, брати відповідні значення з config/env;
- за замовчуванням `--provider=auto`;
- додати unit tests для help і argument validation.

Acceptance criteria:

- `python -m market_data_service.main candles:get --help` працює;
- без `--period` команда повертає non-zero exit code;
- invalid period format повертає non-zero exit code;
- `--provider` приймає тільки `auto`, `binance`, `bybit`.

### Етап 3. Зафіксувати canonical timeframe-и

Мета: зробити єдину внутрішню модель timeframe-ів для всіх provider-ів.

Завдання:

- зафіксувати canonical values `1h`, `4h`, `1d`;
- додати validation для configured `MARKET_DATA_TIMEFRAMES`;
- додати tests для valid/invalid timeframe values.

Acceptance criteria:

- вся application logic працює з `1h`, `4h`, `1d`;
- біржові interval-и не протікають у public CLI/config.

### Етап 4. Виділити provider port для candles

Мета: підготувати спільний контракт для Binance і Bybit.

Завдання:

- виділити спільний provider port для fetch candles;
- прибрати Binance-specific типи з application-facing контракту;
- зафіксувати normalized candle model.

Acceptance criteria:

- provider port не містить Binance-specific типів;
- майбутній Bybit adapter може реалізувати той самий port.

### Етап 5. Перевести Binance adapter на provider port

Мета: привести існуючий Binance adapter до нового контракту без зміни поведінки збору.

Завдання:

- зробити Binance adapter implementation нового provider port;
- замапити Binance intervals: `1h -> 1h`, `4h -> 4h`, `1d -> 1d`;
- нормалізувати output у спільну candle model;
- додати tests для mapping і normalized output.

Acceptance criteria:

- Binance adapter приймає `1h`, `4h`, `1d`;
- normalized output не залежить від raw Binance response format.

### Етап 6. Додати Bybit candles adapter

Мета: додати друге джерело candles для symbols, яких немає на Binance.

Завдання:

- додати Bybit public candles adapter;
- замапити Bybit intervals: `1h -> 60`, `4h -> 240`, `1d -> D`;
- нормалізувати output у спільну candle model;
- додати unit tests для response mapping і provider errors.

Acceptance criteria:

- Bybit adapter повертає normalized candles для `1h`, `4h`, `1d`;
- Binance adapter і Bybit adapter реалізують один provider port.

### Етап 7. Додати pagination і timeout behavior для provider adapters

Мета: зробити fetch candles стабільним для range-ів, більших за provider limit.

Завдання:

- додати pagination support для Binance adapter;
- додати pagination support для Bybit adapter;
- уніфікувати timeout handling;
- уніфікувати provider error codes;
- додати tests для pagination, timeout і transient errors.

Acceptance criteria:

- обидва adapter-и можуть отримувати великий range через pagination;
- timeout і transient errors повертаються контрольовано.

### Етап 8. Додати provider priority config

Мета: явно описати порядок вибору джерел даних.

Завдання:

- додати `MARKET_DATA_PROVIDER_PRIORITY=binance,bybit`;
- додати validation provider names;
- зафіксувати default priority Binance -> Bybit;
- додати config tests.

Acceptance criteria:

- invalid provider name не проходить validation;
- якщо env не заданий, використовується Binance -> Bybit.

### Етап 9. Додати persistence для provider availability

Мета: зберігати інформацію про доступність symbol на конкретному provider-і.

Завдання:

- додати DB schema/migration для provider availability;
- додати statuses `SUPPORTED`, `UNSUPPORTED`, `TEMPORARY_ERROR`;
- додати `next_check_at`;
- додати repository tests.

Acceptance criteria:

- availability зберігається для конкретної пари `source/requested_symbol`;
- `next_check_at` доступний для retry/recheck logic.

### Етап 10. Реалізувати provider availability cache rules

Мета: не робити зайві network requests до provider-а, де symbol вже відомо відсутній.

Завдання:

- додати TTL config:

```bash
MARKET_DATA_PROVIDER_AVAILABILITY_TTL_HOURS=24
MARKET_DATA_UNSUPPORTED_SYMBOL_RECHECK_HOURS=24
```

- реалізувати правило `UNSUPPORTED` до `next_check_at`;
- реалізувати коротший retry/backoff для `TEMPORARY_ERROR`;
- додати tests для cache hit/cache expired behavior.

Acceptance criteria:

- `UNSUPPORTED` кешується тільки для конкретної пари `source/requested_symbol`;
- `TEMPORARY_ERROR` не трактується як `UNSUPPORTED`;
- актуальний `UNSUPPORTED` блокує повторний request тільки до того самого provider-а.

### Етап 11. Реалізувати multi-provider symbol resolver

Мета: вибирати active provider для symbol за configured priority.

Завдання:

- реалізувати resolution Binance -> Bybit;
- використовувати provider availability cache;
- не блокувати Bybit через Binance `UNSUPPORTED`;
- контрольовано повертати unresolved result, якщо symbol відсутній всюди;
- додати resolver tests.

Acceptance criteria:

- якщо symbol є на Binance, active provider - `BINANCE_SPOT`;
- якщо symbol відсутній на Binance, але є на Bybit, active provider - `BYBIT_SPOT`;
- якщо symbol відсутній на всіх provider-ах, він пропускається контрольовано і не валить сервіс.

### Етап 12. Підключити resolver до `candles:get --provider auto`

Мета: зробити ручний fetch candles сумісним із multi-provider правилами.

Завдання:

- у режимі `--provider auto` використовувати resolver;
- у режимі `--provider binance` або `--provider bybit` обходити priority list;
- додати tests для explicit provider і auto provider.

Acceptance criteria:

- `candles:get --provider auto` бере Binance для supported Binance symbols;
- `candles:get --provider auto` бере Bybit для symbols, яких немає на Binance, але є на Bybit;
- explicit provider mode не перемикається на інший provider.

### Етап 13. Реалізувати `candles:get` fetch service

Мета: виконувати manual fetch candles за lookback period і записувати отримані candles у БД як тестові/історичні дані без запуску event-driven bot flow.

Завдання:

- додати service для fetch за `[now - period, now)`;
- підтримати explicit `--symbols` і `--timeframes`;
- підтримати config mode, якщо symbols/timeframes не передані;
- записувати отримані candles у БД через idempotent upsert;
- не створювати market-data outbox events для цього manual load path;
- не публікувати Redis events;
- не запускати торгових ботів;
- виводити результат у JSON summary;
- додати tests для period/range validation, DB write/upsert, JSON output і відсутності event/outbox side effects.

Acceptance criteria:

- команда отримує candles за period-derived range `[now - period, now)`;
- команда записує отримані candles у БД;
- команда без `--symbols` використовує configured symbols;
- команда без `--timeframes` використовує configured timeframes;
- повторний запуск не створює duplicate candles;
- команда не створює outbox/Redis events;
- команда не запускає ботів.

### Етап 14. Формалізувати market-data event model

Мета: зробити подію, яку безпечно споживає `bot-platform-service`.

Завдання:

- додати typed event model `MarketDataCandlesCollectedEvent`;
- додати `contract_version`;
- додати стабільний `idempotency_key`;
- зафіксувати range semantics `[from, to)`;
- додати tests для event payload.

Acceptance criteria:

- event містить source/symbol/provider_symbol/timeframe/range/batch/snapshot/idempotency;
- `from` inclusive, `to` exclusive.

### Етап 15. Додати transactional outbox для market-data events

Мета: гарантувати, що event не загубиться після успішного запису candles.

Завдання:

- створювати outbox event record в одній DB transaction з candles/batch/snapshot;
- Redis publish виконувати тільки після commit;
- додати retry для pending outbox events;
- додати tests для failed DB write і failed Redis publish.

Acceptance criteria:

- outbox event record не створюється окремо після commit candles;
- Redis event не публікується при failed DB write;
- якщо Redis publish failed, outbox event лишається pending/retry.

### Етап 16. Додати retention config для Market Data events

Мета: не накопичувати Redis/outbox event data без обмеження.

Завдання:

- додати `MARKET_DATA_REDIS_EVENT_RETENTION_DAYS=2`;
- додати `MARKET_DATA_OUTBOX_RETENTION_DAYS=5`;
- зафіксувати, що 5-денний cleanup не застосовується до `_market_data.market_candles`;
- додати config validation tests.

Acceptance criteria:

- invalid retention values не проходять validation;
- default retention: Redis events - 2 дні, outbox DB records - 5 днів;
- candles не видаляються cleanup-ом службових event records.

### Етап 17. Додати cleanup для Market Data outbox

Мета: видаляти старі terminal outbox records із БД.

Завдання:

- додати cleanup service для terminal status `PUBLISHED`;
- не видаляти `PENDING` і retryable `FAILED`;
- виконувати cleanup batch-ами;
- додати metrics/logs для кількості видалених записів;
- додати tests для retention cutoff і non-terminal protection.

Acceptance criteria:

- terminal outbox records старші за 5 днів видаляються;
- non-terminal records не видаляються;
- cleanup повторно запускається без побічних ефектів.

### Етап 18. Додати Redis Stream trimming для Market Data events

Мета: не тримати старі delivery events у Redis довше 2 днів.

Завдання:

- додати trimming policy для `market-data-events`;
- якщо Redis/time-based trimming недоступний у вибраному клієнті, використовувати bounded `MAXLEN`, розрахований під 2-денне recovery window з documented throughput assumption і safety margin;
- не trim-ити events раніше, ніж їх може прочитати consumer у межах recovery window;
- додати tests або integration smoke для trimming configuration.

Acceptance criteria:

- Redis Stream не росте без обмеження;
- event delivery recovery window становить приблизно 2 дні за documented throughput assumption;
- trimming не видаляє нові events.

### Етап 19. Створити `CandleCollectionService`

Мета: прибрати залежність production flow від трьох окремих CLI-команд.

Завдання:

- додати application service без підключення до long-running loop;
- перенести потрібну orchestration logic зі старих `scheduler:run-once`, `sync:run-next`, `outbox:publish-once`;
- service має обробляти configured/resolved source/symbol/timeframe ranges;
- service має використовувати provider resolver;
- додати unit tests для orchestration без real provider calls.

Acceptance criteria:

- service може виконати один collection tick без CLI;
- старі CLI-команди поки можуть залишатися до етапу видалення.

### Етап 20. Додати bootstrap range logic

Мета: на першому запуску синхронізувати історію за останні 2 роки.

Завдання:

- додати `MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS=2`;
- додати `MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK`;
- додати `MARKET_DATA_PROVIDER_MAX_CONCURRENCY`;
- для першого запуску кожної `source/symbol/timeframe` пари обчислювати range за 2 роки;
- розбивати bootstrap range на chunks і зберігати progress між ітераціями;
- дозволити bootstrap завершуватися поступово протягом багатьох ітерацій, щоб не впиратися в provider rate limits;
- якщо provider має менше історії, приймати provider-limited bootstrap;
- позначати такий range як bootstrap/backfill mode;
- зафіксувати, що bootstrap/backfill mode не створює market-data events;
- додати tests для full bootstrap, limited bootstrap і resume bootstrap progress.

Acceptance criteria:

- перший запуск синхронізує останні 2 роки, якщо provider має такі дані;
- bootstrap може виконуватися частинами і продовжуватися з останнього успішного chunk;
- bootstrap не перевищує configured provider concurrency/rate limits;
- менша provider history не вважається помилкою;
- bootstrap/backfill result не створює outbox event.

### Етап 21. Додати incremental catch-up range logic

Мета: після downtime добирати всі пропущені candles без gaps.

Завдання:

- визначати останню збережену closed candle;
- будувати наступний event range після останньої збереженої closed candle до останньої безпечної closed candle;
- не запитувати open/incomplete candle;
- додати tests для downtime gap recovery.

Acceptance criteria:

- наступні ітерації добирають пропущений range;
- open candle не записується як closed candle.

### Етап 22. Додати DB write/upsert для candles і snapshots

Мета: зробити запис candles idempotent.

Завдання:

- записувати candles через unique key/upsert;
- записувати batch/snapshot metadata;
- не створювати дублікати при повторному запуску;
- додати repository/service tests.

Acceptance criteria:

- повторний collection tick не створює duplicate candles;
- snapshot/batch metadata відповідають фактично записаному range.

### Етап 23. Додати DB indexes для candles і snapshots

Мета: забезпечити швидке отримання candles, event snapshot by id і latest snapshot для readiness/manual diagnostics.

Завдання:

- перевірити, що `_market_data.market_candles` має unique/index `(source, canonical_symbol, timeframe, open_time)`;
- прибрати або залишити тільки обґрунтовані дублюючі candle indexes після `EXPLAIN ANALYZE`;
- додати index `market_snapshots_latest_complete_idx` для latest complete snapshot lookup;
- перевірити, що `market_snapshot_candles` читає membership через `snapshot_id` без full scan;
- додати migration test або smoke test для наявності індексів.

Acceptance criteria:

- range lookup candles використовує index по `source/canonical_symbol/timeframe/open_time`;
- `get_latest_complete_snapshot` використовує `market_snapshots_latest_complete_idx`;
- snapshot candles read не виконує full scan по всіх snapshots;
- `EXPLAIN ANALYZE` на representative queries не показує sequential scan на великих таблицях.

### Етап 24. Підключити transactional event creation до collection service

Мета: після успішного запису live/incremental candles створювати event для bot-platform.

Завдання:

- у межах DB transaction записувати candles, batch/snapshot і outbox event тільки для live/incremental range;
- для bootstrap/backfill range записувати candles і batch/snapshot без outbox event;
- після commit публікувати Redis event тільки якщо outbox event був створений;
- використовувати стабільний idempotency key;
- додати tests для duplicate-safe publish;
- додати tests, що bootstrap/backfill не створює event.

Acceptance criteria:

- live/incremental `collect --once` пише candles/snapshot/batch/outbox event;
- bootstrap/backfill `collect --once` пише candles/snapshot/batch без outbox event;
- після commit публікується Redis event тільки для live/incremental range;
- повторний запуск не створює неконтрольовані duplicate events.

### Етап 25. Підключити `collect --once` до `CandleCollectionService`

Мета: замінити ручний ingest chain однією one-shot командою.

Завдання:

- CLI `collect --once` має викликати `CandleCollectionService`;
- команда має обробляти всі configured symbols/timeframes;
- команда має повертати non-zero тільки для terminal failure;
- додати CLI/service integration tests.

Acceptance criteria:

- `collect --once` збирає candles;
- при першому bootstrap запуску `collect --once` не створює event;
- після завершеного bootstrap live/incremental `collect --once` створює event;
- існуючий Docker runtime smoke можна переписати на `collect --once`.

### Етап 26. Додати long-running loop для `collect`

Мета: `collect` без `--once` має залишатися running і запускати collection ticks.

Завдання:

- додати loop з poll interval;
- на кожній ітерації синхронізувати всі configured symbols/timeframes;
- додати env vars:

```bash
MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS=30
MARKET_DATA_COLLECT_ON_START=true
MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK=100
MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS=2
MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK=20
MARKET_DATA_PROVIDER_MAX_CONCURRENCY=3
MARKET_DATA_PROVIDER_PRIORITY=binance,bybit
MARKET_DATA_TIMEFRAMES=1h,4h,1d
```

- додати tests для one tick і repeated ticks.

Acceptance criteria:

- `collect` залишається running;
- збирає candles без ручного `sync:run-next`.

### Етап 27. Додати graceful shutdown для `collect`

Мета: коректно завершувати long-running процес.

Завдання:

- обробляти SIGTERM/SIGINT;
- не переривати DB transaction посередині;
- закривати DB/Redis/provider clients;
- додати smoke або unit tests для shutdown path, якщо можливо.

Acceptance criteria:

- `docker stop` завершує процес cleanly;
- немає partial commit без контрольованого batch status.

### Етап 28. Додати health/readiness/metrics для `collect`

Мета: зробити сервіс керованим у Docker/production-like середовищі.

Завдання:

- long-running `collect` має запускати поточний HTTP server у тому самому процесі або через той самий runtime container;
- readiness має перевіряти DB/Redis;
- metrics мають показувати collection success/failure counters;
- `healthcheck` має перевіряти HTTP readiness endpoint, який піднятий long-running `collect`;
- `collect --once` не потребує HTTP health/readiness, бо це short-lived command;
- `serve` після переходу не має бути production entrypoint; якщо staged rollout більше його не використовує, CLI route треба видалити разом зі старими ingest commands.

Acceptance criteria:

- readiness показує non-ready, якщо DB/Redis недоступні;
- long-running `collect` має production healthcheck path;
- Docker healthcheck через `python -m market_data_service.main healthcheck` працює проти HTTP readiness long-running `collect`;
- у фінальному стані основна команда - `collect`, а `serve` не використовується в Docker/README.

### Етап 29. Bot Platform market-data event consumer skeleton

Мета: додати consumer без запуску bot adapters.

Завдання:

- додати Redis Stream consumer для market-data events;
- додати settings:

```bash
BOT_PLATFORM_MARKET_DATA_EVENTS_ENABLED=true
BOT_PLATFORM_MARKET_DATA_EVENTS_REDIS_URL=redis://redis:6379/0
BOT_PLATFORM_MARKET_DATA_EVENTS_STREAM=market-data-events
BOT_PLATFORM_MARKET_DATA_EVENTS_CONSUMER_GROUP=bot-platform
```

- валідувати event schema/version;
- логувати recognized events;
- не запускати ботів у цьому етапі.

Acceptance criteria:

- consumer читає events і ACK-ає валідні no-op events;
- invalid events не валять процес і переводяться у контрольований terminal/no-op path без нескінченного retry loop;
- Redis unavailable не ламає HTTP admin API.

### Етап 30. Додати event idempotency у Bot Platform consumer

Мета: зробити повторне читання events безпечним.

Завдання:

- зберігати processed event/idempotency state;
- не обробляти той самий event двічі;
- додати tests для duplicate event delivery.

Acceptance criteria:

- повторне читання event не створює duplicate processing;
- ACK/retry behavior контрольований.

### Етап 31. Додати retention config для Bot Platform event data

Мета: не накопичувати processed/idempotency і event audit records без обмеження.

Завдання:

- додати `BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS=5`;
- додати `BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS=5`;
- зафіксувати, що cleanup не видаляє business records, потрібні для користувацької історії;
- додати config validation tests.

Acceptance criteria:

- default retention для processed/idempotency records - 5 днів;
- invalid retention values не проходять validation;
- retention policy не видаляє active bot runs.

### Етап 32. Додати cleanup для Bot Platform event data

Мета: видаляти старі службові event records після завершення replay window.

Завдання:

- видаляти processed/idempotency records старші за 5 днів;
- видаляти raw event/audit delivery records старші за 5 днів, якщо вони не потрібні для active incident/retry;
- не видаляти active bot runs/signals, які є business history;
- виконувати cleanup batch-ами;
- додати tests для cutoff і protection active records.

Acceptance criteria:

- службові event records старші за 5 днів видаляються;
- active/non-terminal processing records не видаляються;
- cleanup idempotent.

### Етап 33. Event -> enabled bot instances

Мета: навчити consumer знаходити ботів для event symbol/timeframe.

Завдання:

- додати query для enabled instances by symbol/timeframe;
- перевіряти mode `signal_only`;
- фільтрувати instances, які не підтримують timeframe;
- додати tests.

Acceptance criteria:

- для event `BTCUSDT/1h` знаходяться тільки відповідні enabled instances;
- paused/disabled instances не запускаються;
- unmatched event ACK-иться як no-op.

### Етап 34. Event -> bot run

Мета: запускати ботів після market-data event.

Завдання:

- викликати існуючий manual run/orchestration service з event-derived command;
- передавати у bot run snapshot з `snapshot_id` market-data event, а не latest snapshot lookup;
- використати idempotency key:

```text
market-data-event:{source}:{symbol}:{timeframe}:{closed_at}:{instance_id}
```

- писати `bot_runs`, `bot_run_events`, `bot_signals`, audit events;
- ACK event після успішної обробки всіх applicable instances або контрольованого terminal result;
- retry для transient errors.

Acceptance criteria:

- одна market-data event створює bot run для кожного matching enabled instance;
- повторне читання event не створює duplicate runs;
- signal-only safety збережена.

### Етап 35. Оновити Docker Compose для Market Data

Мета: Docker має запускати `market-data-service` через `collect`.

Завдання:

- `market_data` command замінити на `collect`;
- прокинути collect env vars;
- прокинути retention env vars для Redis/outbox cleanup;
- увімкнути Redis stream для market-data events;
- видалити Docker jobs старого market-data ingest flow.

Acceptance criteria:

- `docker compose up market_data` запускає `collect`;
- candles пишуться у БД;
- market-data events публікуються для incremental/live ranges після завершеного bootstrap.

### Етап 36. Оновити Docker Compose для Bot Platform runner

Мета: Docker має запускати consumer, який реагує на market-data events.

Завдання:

- `bot_platform_runner` або окремий worker має запускати market-data event consumer;
- прокинути consumer env vars;
- прокинути retention env vars для idempotency/audit cleanup;
- перевірити, що HTTP API і runner можуть працювати незалежно;
- додати Docker smoke для event consumption.

Acceptance criteria:

- `docker compose up bot_platform bot_platform_runner` запускає consumer;
- bot-platform запускає matching bots по market-data events;
- Redis unavailable не валить HTTP API.

### Етап 37. Видалити старі CLI routes

Мета: прибрати user-facing команди, які більше не є частиною production workflow.

Завдання:

- видалити CLI routes для `scheduler:run-once`, `sync:run-next`, `outbox:publish-once`;
- перенести корисні CLI tests на `collect` і `candles:get`;
- перевірити, що Docker/README більше не посилаються на старі команди.

Acceptance criteria:

- `rg "scheduler:run-once|sync:run-next|outbox:publish-once"` не знаходить user-facing docs/compose references;
- test suite проходить.

### Етап 38. Видалити непотрібний ingest code і оновити README

Мета: прибрати dead code і зафіксувати актуальний спосіб запуску.

Завдання:

- видалити dead code, якщо він більше не викликається production/test шляхом;
- видалити Docker jobs, які більше не використовуються;
- видалити README-описи старих команд;
- залишити в README тільки `collect`, `candles:get`, migrations/healthcheck/ops commands.
- описати retention: Redis events 2 дні, службові DB event records 5 днів, candles не видаляються цим cleanup-ом.

Acceptance criteria:

- у README лишаються тільки актуальні команди;
- dead code не має production/test references;
- test suite проходить.

### Етап 39. Full smoke Market Data flow

Мета: підтвердити, що market-data частина нового flow працює з нуля.

Завдання:

- підняти PostgreSQL/Redis;
- застосувати migrations;
- запустити `market_data collect --once` або long-running collect з коротким interval для bootstrap range;
- перевірити candles/snapshots у БД;
- перевірити, що bootstrap range не створив Redis event;
- запустити наступний incremental/live collect range;
- перевірити Redis event для incremental/live range;
- перевірити, що cleanup не видаляє candles;
- перевірити, що старі terminal outbox records видаляються за retention policy;
- перевірити idempotent повторний запуск.

Acceptance criteria:

- candles і snapshots створюються;
- bootstrap range не публікує Redis event;
- incremental/live range публікує Redis event;
- повторний smoke не створює неконтрольовані дублікати.

### Етап 40. Full smoke Event-driven Bot Platform flow

Мета: підтвердити end-to-end сценарій.

Завдання:

- підняти повний Docker flow;
- запустити `bot_modules_sync`;
- запустити market-data collect і завершити bootstrap без bot runs;
- запустити наступний incremental/live collect range;
- перевірити, що `bot-platform-service` запустив bot run;
- перевірити `bot_runs`, `bot_signals`;
- перевірити cleanup processed/idempotency records;
- перевірити idempotent повторний запуск.

Acceptance criteria:

- один smoke script проходить з нуля;
- повторний smoke не створює неконтрольовані дублікати;
- legacy standalone bot services не стартують;
- старі ingest CLI-команди не потрібні для smoke.

## 5. Команди у фінальному стані

### 5.1. Market Data

Production collect:

```bash
python -m market_data_service.main collect
```

Ця команда запускає `market-data-service` як long-running сервіс. На кожній ітерації сервіс синхронізує candles для всіх symbols/timeframes з `MARKET_DATA_PROVIDER_SYMBOLS` і `MARKET_DATA_TIMEFRAMES` та записує результат у БД. Перший bootstrap/backfill history sync не створює events. Після завершеного bootstrap сервіс створює events для `bot-platform-service` тільки для live/incremental ranges.

За замовчуванням очікувані timeframe-и:

```bash
export MARKET_DATA_TIMEFRAMES=1h,4h,1d
```

Provider selection:

```bash
export MARKET_DATA_PROVIDER_PRIORITY=binance,bybit
```

Retention:

```bash
export MARKET_DATA_REDIS_EVENT_RETENTION_DAYS=2
export MARKET_DATA_OUTBOX_RETENTION_DAYS=5
```

Якщо symbol, наприклад `HYPEUSDT`, відсутній на Binance, але доступний на Bybit, `collect` має синхронізувати candles для `HYPEUSDT` з Bybit і зафіксувати, що Binance для цього symbol `UNSUPPORTED` до `next_check_at`.

One-shot collect:

```bash
python -m market_data_service.main collect --once
```

Load test candles from active provider into DB without bot events:

```bash
python -m market_data_service.main candles:get \
  --period 100d \
  --symbols HYPEUSDT \
  --timeframes 1h,4h,1d \
  --provider auto
```

Load test candles for configured symbols/timeframes into DB without bot events:

```bash
export MARKET_DATA_PROVIDER_SYMBOLS=BTCUSDT,ETHUSDT,HYPEUSDT
export MARKET_DATA_TIMEFRAMES=1h,4h,1d
export MARKET_DATA_PROVIDER_PRIORITY=binance,bybit

python -m market_data_service.main candles:get \
  --period 1y \
  --provider auto
```

Healthcheck, якщо потрібен Docker healthcheck:

```bash
python -m market_data_service.main healthcheck
```

### 5.2. Bot Platform

HTTP API:

```bash
python -m bot_platform_service.main serve
```

Runner/event consumer:

```bash
python -m bot_platform_service.main runner
```

Retention:

```bash
export BOT_PLATFORM_EVENT_IDEMPOTENCY_RETENTION_DAYS=5
export BOT_PLATFORM_EVENT_AUDIT_RETENTION_DAYS=5
```

Sync bot modules:

```bash
python -m bot_platform_service.main bot:modules:sync
```

## 6. Основні ризики

- Event може бути опублікований до commit candles у БД. Це треба заборонити архітектурно.
- Redis Stream consumer може створювати duplicate bot runs. Потрібна idempotency на рівні bot run.
- `collect` може дублювати candles при повторному запуску. Потрібні unique constraints/upsert behavior.
- Перший запуск синхронізує 2 роки candles для кожної configured пари, тому bootstrap має виконуватися поступово з pagination, rate limits, concurrency limits і persisted progress.
- Деякі symbols можуть мати історію меншу за 2 роки; це має бути provider-limited bootstrap, а не failure/gap.
- Після downtime сервіс має добирати весь пропущений range, інакше зʼявляться gaps у candles.
- Provider pagination/rate limits можуть створити partial ingest. Потрібен контрольований batch status.
- Provider availability cache може застаріти: symbol міг зʼявитися на Binance після `UNSUPPORTED`, тому потрібен TTL і `next_check_at`.
- Якщо один symbol доступний на кількох provider-ах, priority list має однозначно визначати active provider.
- Видалення старих команд може зламати smoke/README/compose, якщо не перенести всі сценарії на `collect`.
- Bot runner не повинен запускати боти на stale snapshots або на snapshot, який не відповідає `snapshot_id` event.
- Latest snapshot lookup без спеціального індексу може стати bottleneck для readiness/manual diagnostics.
- Зайві дублюючі indexes на candles можуть сповільнювати insert/upsert під час bootstrap, тому їх треба перевіряти через `EXPLAIN ANALYZE`, а не додавати без потреби.
- Redis Stream без trimming буде накопичувати старі market-data events у памʼяті, тому retention 2 дні має бути обовʼязковим.
- Outbox/idempotency/audit event records без cleanup будуть рости в БД, тому terminal службові records мають видалятися після 5 днів.
- 5-денний cleanup не можна застосовувати до `_market_data.market_candles`, інакше буде зламано 2-річний bootstrap, gap recovery і роботу ботів з історією.

## 7. Definition of Done

Функціонал вважається реалізованим, коли:

- `collect --once` збирає candles з active provider-а і пише їх у БД;
- `collect` працює як long-running процес;
- перший запуск `collect` синхронізує останні 2 роки candles для кожної configured `source/symbol/timeframe` пари;
- перший bootstrap запуск `collect` не створює market-data events і не запускає ботів;
- bootstrap може виконуватися поступово кількома ітераціями без перевищення provider rate limits;
- bootstrap progress зберігається і дозволяє продовжити синхронізацію після restart;
- market-data events створюються тільки для live/incremental ranges після завершеного bootstrap;
- collect синхронізує timeframe-и `1h`, `4h`, `1d`;
- якщо active provider має менше ніж 2 роки історії для symbol/timeframe, сервіс записує весь доступний provider range без помилки;
- наступні ітерації `collect` добирають усі candles, пропущені після останньої успішно збереженої closed candle;
- provider resolver вибирає Binance для symbols, які є на Binance;
- provider resolver вибирає Bybit для symbols, яких немає на Binance, але які є на Bybit;
- unsupported provider/symbol pairs кешуються і не створюють зайвих network requests до `next_check_at`;
- candles range lookup покритий індексом `source/canonical_symbol/timeframe/open_time`;
- latest complete snapshot lookup покритий індексом `market_snapshots_latest_complete_idx`;
- representative `EXPLAIN ANALYZE` для candles range і latest snapshot lookup не показує sequential scan на великих таблицях;
- `candles:get` отримує candles за період для explicit або configured symbols/timeframes, записує їх у БД і не створює bot-triggering events;
- `candles:get --provider auto` використовує той самий provider resolver, що й `collect`;
- `bot-platform-service` споживає market-data events і запускає matching enabled bots;
- bot runs використовують `snapshot_id` з market-data event, а не випадковий latest snapshot;
- `bot_runs` і `bot_signals` створюються автоматично після market-data event;
- повторний запуск не створює неконтрольовані дублікати;
- Redis Stream market-data events мають 2-денне recovery window через time-based cleanup або bounded `MAXLEN`, розрахований за documented throughput assumption;
- terminal market-data outbox records видаляються з БД після 5 днів;
- bot-platform processed/idempotency і службові event audit records видаляються з БД після 5 днів;
- `_market_data.market_candles` не видаляються retention cleanup-ом для службових event records;
- старі user-facing ingest команди видалені з CLI, Docker Compose і README;
- повний Docker smoke нового flow проходить однією командою.
