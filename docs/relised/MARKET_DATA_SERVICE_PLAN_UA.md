# План винесення production-ready Market Data Service

## 1. Мета

Винести завантаження, перевірку, збереження та публікацію готовності ринкових свічок в окремий спільний Market Data Service, який буде канонічним джерелом market candles і snapshots для майбутніх споживачів.

Сервіс має:

- один раз завантажувати свічки з Binance Spot або іншого джерела;
- зберігати закриті свічки у спільній БД;
- гарантувати повноту, актуальність і відсутність gaps для потрібних lookback-вікон;
- створювати immutable market snapshots;
- публікувати події готовності після commit у БД;
- синхронізувати символи незалежно, не чекаючи завершення sync усіх symbol/timeframe pair;
- бути готовим до майбутнього підключення стратегій, ботів і multi-tenant trading platform, але не виконувати це підключення в межах цього плану.

## 1.1. Директорія Реалізації

Market Data Service реалізовується в окремій директорії:

```text
market-data-service/
```

Стандарти структури, неймінгу, шарів, persistence rules, testing rules і prohibited patterns для цієї директорії описані в:

```text
market-data-service/STANDARDS.md
```

Усі нові файли Market Data Service мають створюватися всередині `market-data-service/` і відповідати цьому стандарту. Код `spot_grid_bot`, `spot-greenwich-bot` і `register-service` не змінюється в межах цього плану. Підключення ботів до snapshots буде окремим пізнішим планом.

## 2. Межі Відповідальності

### Market Data Service відповідає за:

- планування sync jobs для `symbol + timeframe`;
- завантаження свічок із зовнішнього market-data provider;
- нормалізацію timestamps, OHLCV і статусу закритості свічок;
- idempotent insert свічок у БД без дублювання;
- gap detection і backfill;
- freshness/completeness checks;
- створення `market_data_batches` і `market_snapshots`;
- публікацію `CandleBatchReady` після готовності candle snapshot;
- retry, rate-limit handling, observability і audit sync-процесу.

### Market Data Service не відповідає за:

- генерацію торгових сигналів;
- портфельний risk management;
- execution ордерів;
- біржові private API credentials;
- tenant entitlements;
- торгові позиції, balances, open orders або fills;
- готовність стратегій, запуск planning/execution або інтеграцію з ботами.

## 3. Цільова Схема

```text
Scheduler
  -> MarketDataSyncJob(symbol, timeframe, expected_close_time)
  -> Market Data Service
       -> Provider Adapter: Binance Spot
       -> Candle Normalizer
       -> Gap/Freshness Validator
       -> PostgreSQL candles + snapshots
       -> Transactional Outbox
  -> CandleBatchReady(symbol, timeframe, snapshot_id)
  -> Outbox Publisher
       -> broker/queue delivery for future consumers
```

## 3.1. Persistence Stack

Market Data Service має використовувати:

```text
SQLAlchemy Core 2.0
asyncpg
Alembic
PostgreSQL
```

Рішення:

- `SQLAlchemy Core 2.0` використовується як основний query builder і schema expression layer;
- `asyncpg` використовується як async PostgreSQL driver через SQLAlchemy async engine;
- `Alembic` використовується для всіх schema migrations;
- класичний SQLAlchemy ORM не використовується в hot path;
- ad-hoc SQL у business logic заборонений, SQL живе в repository/query modules;
- raw SQL допускається лише там, де SQLAlchemy Core не дає достатнього контролю або продуктивності, наприклад PostgreSQL-specific advisory locks, bulk insert з `ON CONFLICT DO NOTHING` або спеціальні hash queries.

Repository layer має відокремити application use cases від SQL details:

```text
MarketSymbolRepository
ProviderSymbolRepository
CandleRepository
BatchRepository
OutboxRepository
```

Вимоги до persistence layer:

- всі write use cases приймають explicit transaction/session boundary;
- bulk candle insert виконується через SQLAlchemy PostgreSQL dialect `insert().on_conflict_do_nothing()` або контрольований raw SQL;
- advisory locks виконуються в межах тієї самої DB transaction, де приймається рішення про sync;
- Decimal/NUMERIC не конвертується у float;
- Alembic migration є єдиним способом змінювати `_market_data` schema;
- tests мають покривати generated SQL або behavior для critical queries: duplicate-safe candle insert, snapshot membership, outbox idempotency, advisory locks.

## 4. Таймфрейми та Snapshot Requirements

Початково підтримуються:

- `1h`
- `4h`
- `1d`

Для кожного supported timeframe сервіс має вміти:

- визначити UTC close boundary;
- застосувати safety delay;
- синхронізувати закриті свічки;
- перевірити completeness/freshness;
- створити batch, snapshot і `CandleBatchReady`.

Вимоги конкретних стратегій і ботів не описуються в цьому плані. Вони будуть додані в окремому плані інтеграції після стабілізації збору свічок.

## 5. Закриті Свічки

Сервіс має зберігати й публікувати тільки закриті свічки.

Правила:

- `1h` свічка вважається закритою тільки після завершення відповідної UTC-години плюс safety delay;
- `4h` свічка закрита після UTC блоків `00`, `04`, `08`, `12`, `16`, `20` плюс safety delay;
- `1d` свічка закрита після `00:00:00 UTC` наступного дня плюс safety delay;
- поточна незакрита свічка не використовується для snapshots;
- provider response має проходити перевірку `open_time`, `close_time`, interval duration і monotonic ordering.

Рекомендовані safety delays:

```text
1h: 10-30 seconds
4h: 15-45 seconds
1d: 30-90 seconds
```

Значення мають бути конфігурованими.

## 6. Канонічна Модель Свічки

Свічки не зберігаються в таблицях виду `<symbol>_<timeframe>`.

Production model:

- одна логічна модель `market_candles`;
- обов'язковий PostgreSQL partitioned parent table `market_candles`;
- фізично окрема partition table для кожного supported timeframe;
- у кожній timeframe-таблиці зберігаються свічки всіх символів;
- додавання нового символу не потребує Alembic migration;
- додавання нового timeframe потребує явної migration і оновлення scheduler/freshness configuration.

Початкові physical tables:

```text
_market_data.market_candles_1h
_market_data.market_candles_4h
_market_data.market_candles_1d
```

Production storage має бути реалізований як PostgreSQL partitioned parent table:

```text
_market_data.market_candles PARTITION BY LIST (timeframe)
  -> _market_data.market_candles_1h FOR VALUES IN ('1h')
  -> _market_data.market_candles_4h FOR VALUES IN ('4h')
  -> _market_data.market_candles_1d FOR VALUES IN ('1d')
```

Application/repository layer працює з parent table `market_candles`, а не будує table name з symbol. PostgreSQL сам маршрутизує rows у timeframe partitions за значенням `timeframe`. Це гарантує глобально однозначний `candle_id` для snapshot membership і не потребує `candle_table` у `market_snapshot_candles`.

```text
market_candles
  candle_id
  source                 -- BINANCE_SPOT
  canonical_symbol       -- ETH/USDT
  provider_symbol        -- ETHUSDT
  timeframe              -- 1h | 4h | 1d
  open_time              -- timestamptz
  close_time             -- timestamptz
  open                   -- numeric
  high                   -- numeric
  low                    -- numeric
  close                  -- numeric
  volume                 -- numeric
  quote_volume           -- numeric nullable
  taker_buy_base_volume  -- numeric nullable, Binance kline field або trade aggregation
  taker_buy_quote_volume -- numeric nullable, Binance kline field або trade aggregation
  taker_sell_base_volume -- numeric nullable, derived або trade aggregation
  taker_sell_quote_volume -- numeric nullable, derived або trade aggregation
  trades_count           -- integer nullable
  is_closed              -- boolean
  provider_payload_hash  -- text
  inserted_at
  updated_at

unique(source, canonical_symbol, timeframe, open_time)
```

Усі ціни й обсяги зберігаються як `NUMERIC`, не `float`.

Недублювання свічок гарантується на рівні БД через unique constraint:

```text
unique(source, canonical_symbol, timeframe, open_time)
```

Запис candles виконується тільки через idempotent insert:

```text
insert ... on conflict(source, canonical_symbol, timeframe, open_time) do nothing
```

Повторний sync того самого `source + canonical_symbol + timeframe + open_time` має пропустити вже існуючий row без помилки. Звичайний sync не перезаписує OHLCV уже збереженої свічки.

Якщо provider пізніше повернув інші OHLCV для вже існуючої закритої свічки, це не має оброблятися як частина звичайного sync. Для таких випадків потрібен окремий audited correction/backfill flow із явно описаною versioning policy.

## 6.1. Symbol Registry

Canonical symbol mapping не повинен бути розкиданий по adapter-ах, workers або майбутніх consumers.

Потрібні таблиці:

```text
market_symbols
  canonical_symbol       -- ETH/USDT
  base_asset             -- ETH
  quote_asset            -- USDT
  status                 -- ACTIVE | PAUSED | DELISTED
  created_at
  updated_at

provider_symbols
  source                 -- BINANCE_SPOT
  canonical_symbol       -- ETH/USDT
  provider_symbol        -- ETHUSDT
  status                 -- TRADING | HALTED | DELISTED
  supported_timeframes   -- 1h,4h,1d або normalized child table
  min_available_time
  max_backfill_days
  metadata_json
  updated_at

unique(source, provider_symbol)
unique(source, canonical_symbol)
```

Market Data Service читає provider mapping тільки з registry. Якщо mapping відсутній, stale або має non-trading status, sync для цього symbol блокується.

## 7. Market Data Batch

Кожна sync-операція створює batch.

```text
market_data_batches
  id
  source
  canonical_symbol
  timeframe
  requested_from
  requested_to
  expected_close_time
  status                 -- RUNNING | COMPLETE | FAILED | INCOMPLETE
  outbox_status          -- NOT_CREATED | PENDING | PUBLISHED | FAILED
  rows_fetched
  rows_inserted
  rows_skipped_duplicate
  rows_hash_mismatch
  gap_count
  first_open_time
  last_close_time
  error_code
  error_message_redacted
  started_at
  completed_at
```

Batch переходить у `COMPLETE` тільки якщо:

- provider request завершився успішно;
- всі нові candles вставлені в БД;
- існуючі duplicate candles безпечно пропущені через `ON CONFLICT DO NOTHING`;
- hash mismatch для вже існуючих candles не ігнорується: batch має записати audit/metric і не перезаписувати OHLCV у normal sync;
- немає gaps у required lookback;
- остання потрібна свічка закрита;
- snapshot створено;
- outbox event записано в тій самій транзакції.

`COMPLETE` означає, що дані та snapshot валідні. Це не означає, що подія вже доставлена в message broker. Стан доставки відстежується через `outbox_status` і таблицю `outbox_events`.

## 8. Market Snapshot

Snapshot фіксує конкретну версію market data, яку майбутні consumers можуть відтворити за `snapshot_id`.

```text
market_snapshots
  id
  source
  canonical_symbol
  timeframe
  last_closed_candle_time
  lookback_start_time
  lookback_end_time
  candle_count
  data_hash
  batch_id
  completeness_status    -- COMPLETE | INCOMPLETE | STALE | GAP_DETECTED
  created_at
```

`data_hash` рахується з ordered candle ids/hash values. Це дозволяє audit і повторюваність downstream reads.

Майбутні consumers отримують не "останні дані", а `snapshot_id` або набір `snapshot_id`.

Snapshot має бути відтворюваним навіть після майбутнього backfill або корекції candle rows. Для цього потрібна таблиця membership:

```text
market_snapshot_candles
  snapshot_id
  candle_id
  ordinal
  candle_hash_at_snapshot

primary key(snapshot_id, ordinal)
unique(snapshot_id, candle_id)
```

Snapshot reader читає candles через `market_snapshot_candles`, а не через "останній range" у `market_candles`.

`candle_id` посилається на row parent table `_market_data.market_candles`. Оскільки `market_candles` є partitioned parent table, `candle_id` лишається глобально однозначним для всіх timeframe partitions.

Якщо provider пізніше повернув інші OHLCV для вже існуючої свічки, звичайний sync пропускає цей row через `ON CONFLICT DO NOTHING` і не змінює історичний snapshot. Будь-яка корекція вже збережених OHLCV має йти тільки через окремий audited correction/backfill flow із явно описаною storage/versioning policy, після чого створюється новий snapshot із новим `data_hash`.

Для першої версії достатньо:

- `market_candles` містить insert-only canonical candle rows для normal sync;
- `market_snapshot_candles` фіксує `candle_id` і `candle_hash_at_snapshot`;
- якщо `provider_payload_hash` для існуючої свічки відрізняється, normal sync записує audit/metric і пропускає row без помилки;
- новий snapshot через змінені OHLCV дозволений тільки після audited correction/backfill flow.

## 9. Gap Detection і Backfill

Для кожного `symbol + timeframe` сервіс має перевіряти:

- відсутність пропущених intervals;
- правильний interval duration;
- monotonic open_time;
- дублікати;
- чи є остання required candle;
- чи не застарілий snapshot.

Якщо gaps знайдені:

1. batch отримує `INCOMPLETE`;
2. створюється backfill job;
3. `CandleBatchReady` для incomplete range не публікується;
4. metric `market_data_gap_count` збільшується;
5. після успішного backfill створюється новий complete snapshot.

Incomplete або stale data не може отримати ready snapshot event.

## 9.1. Freshness Policy

`STALE` має визначатися формально для кожного timeframe.

Початкові production defaults:

```text
1h max_snapshot_lag: 1 closed candle + 5 minutes
4h max_snapshot_lag: 1 closed candle + 10 minutes
1d max_snapshot_lag: 1 closed candle + 30 minutes
```

Для першої версії freshness policy використовується тільки для оцінки snapshot/batch status. Cross-timeframe rules для стратегій не входять у цей план.

## 10. Події

### 10.1. CandleBatchReady

Публікується після commit batch і snapshot.

```text
event_id
event_type = CandleBatchReady
occurred_at
source
canonical_symbol
timeframe
batch_id
snapshot_id
last_closed_candle_time
completeness_status
idempotency_key
snapshot_version
```

`idempotency_key`:

```text
source + canonical_symbol + timeframe + last_closed_candle_time + snapshot_version
```

`snapshot_id` не входить у logical idempotency key як випадковий UUID. Повторний retry того самого завершеного sync не повинен створювати нову logical подію.

Правила версіонування:

- перший complete snapshot для `source + symbol + timeframe + close_time` має `snapshot_version = 1`;
- retry без зміни candle data повертає той самий snapshot або не створює нову подію;
- backfill або provider correction, що змінює `data_hash`, створює `snapshot_version + 1` і новий `CandleBatchReady`;
- downstream consumers використовують `snapshot_version` для розуміння, що це нова версія market data, а не duplicate delivery.

## 11. Transactional Outbox

Події не публікуються напряму з memory після запису candles.

Потрібна таблиця:

```text
outbox_events
  id
  event_type
  aggregate_type
  aggregate_id
  payload_json
  idempotency_key
  status                 -- PENDING | PUBLISHED | FAILED
  attempts
  next_attempt_at
  created_at
  published_at

unique(event_type, idempotency_key)
```

Flow:

1. у транзакції записати candles;
2. записати batch;
3. записати snapshot;
4. записати outbox event;
5. commit;
6. outbox publisher доставляє подію в Redis Streams/RabbitMQ;
7. consumer-и працюють ідемпотентно.

## 12. Черги та Locks

Потрібні окремі job types:

```text
MarketDataSyncRequested
MarketDataBackfillRequested
OutboxPublishRequested
```

Locks:

- `source + symbol + timeframe` lock для sync одного потоку;
- PostgreSQL advisory lock у межах SQLAlchemy async transaction є default для sync critical section;
- Redis lock з fencing token допускається лише для cross-service orchestration, якщо DB lock недостатній;
- duplicate jobs мають coalesce-итись;
- retry не повинен створювати дублікати candles, snapshots або events.

Рекомендація для старту:

- PostgreSQL як source of truth;
- Redis Streams або BullMQ/Redis для jobs;
- PostgreSQL advisory locks через SQLAlchemy Core/asyncpg для критичних DB-секцій;
- outbox publisher окремим worker-процесом.

## 13. Rate Limit і Provider Adapter

Provider adapter не має бути розкиданий по workers або майбутніх consumers.

Інтерфейс:

```text
fetch_closed_candles(symbol, timeframe, from_time, to_time) -> list[Candle]
get_server_time()
get_rate_limit_state()
```

Rate limit policy:

- глобальний ліміт на provider;
- окремі ліміти на endpoint group;
- exponential backoff для retryable errors;
- circuit breaker при масових provider failures;
- jitter для scheduled jobs;
- metrics для `429`, timeouts, partial responses.

Якщо Binance недоступний:

- нові snapshots не створюються;
- stale data не отримує ready snapshot event;
- поведінка майбутніх trading/risk consumers поза межами цього плану.

## 14. Database Schema Ownership

Market data таблиці мають належати trading service, але бути окремим module/schema.

Schema definitions мають бути представлені в SQLAlchemy Core metadata, а зміни застосовуються тільки через Alembic migrations. Ручне редагування production schema поза Alembic заборонене.

Рекомендована схема:

```text
_market_data.market_candles
_market_data.market_candles_1h
_market_data.market_candles_4h
_market_data.market_candles_1d
_market_data.market_symbols
_market_data.provider_symbols
_market_data.market_data_batches
_market_data.market_snapshots
_market_data.market_snapshot_candles
_market_data.sync_jobs
_market_data.outbox_events
_market_data.provider_health
```

Поточну `_candles_trading_data` можна використати як одне з джерел backfill, але перемикання ботів/readers не входить у цей план.

1. створити нову canonical schema;
2. backfill із Binance або існуючих таблиць;
3. перевірити completeness/hash consistency;
4. залишити legacy schema без змін до окремого плану інтеграції ботів.

## 15. Observability

Метрики:

```text
market_data_sync_duration_seconds
market_data_sync_rows_fetched_total
market_data_sync_rows_inserted_total
market_data_sync_rows_skipped_duplicate_total
market_data_gap_count
market_data_snapshot_age_seconds
market_data_batch_status_total
market_data_provider_errors_total
market_data_provider_rate_limited_total
market_data_outbox_lag_seconds
```

Логи мають містити:

- `source`
- `canonical_symbol`
- `timeframe`
- `batch_id`
- `snapshot_id`
- `last_closed_candle_time`
- `status`
- `gap_count`
- `correlation_id`

Alerts:

- snapshot stale для configured timeframe;
- gap count > 0;
- provider errors вище threshold;
- outbox lag вище threshold;
- sync jobs stuck у `RUNNING`.

## 16. Failure Modes

### Provider timeout

- retry з backoff;
- batch `FAILED` після max attempts;
- no `CandleBatchReady` event.

### Partial data

- batch `INCOMPLETE`;
- backfill job;
- no ready snapshot event.

### Duplicate job

- lock або unique key не дає паралельного sync;
- повторний job завершується no-op.

### DB commit failed

- подія не створена;
- retry job може повторити sync;
- `ON CONFLICT DO NOTHING` гарантує ідемпотентність без помилки при duplicate candle.

### Outbox publish failed

- event лишається `PENDING`;
- publisher retry;
- consumers ідемпотентні.

## 17. Security

- Market Data Service не має доступу до private exchange API keys.
- Не логувати credentials або user-specific trading secrets.
- Якщо provider потребує API key для market data, зберігати його окремо від execution credentials.
- RBAC: тільки service role може писати market data таблиці.
- Read access для майбутніх consumers тільки до committed snapshots.
- Admin/API debug endpoints не повинні показувати raw provider secrets.

## 18. Етапи Реалізації

Етапи нижче навмисно дрібні. Кожен етап має залишати систему в робочому стані, не ламати поточні live/maintenance flows і мати окрему перевірку готовності.

### Етап 0. Baseline без runtime інтеграції

Мета: зафіксувати поточну поведінку перед винесенням data layer.

- зафіксувати поточні джерела candle reads/writes як baseline для майбутнього backfill/порівняння;
- визначити initial symbol universe і supported timeframes;
- зафіксувати правило: новий сервіс не змінює runtime існуючих застосунків;
- описати smoke scenarios для `1h`, `4h`, `1d`.

Готовність:

- є список current candle dependencies;
- новий шлях ще не використовується live runtime;
- rollback path не потребує schema rollback.

### Етап 1. `_market_data` schema shell

Мета: створити окрему схему без зміни runtime поведінки існуючих застосунків.

- додати SQLAlchemy Core metadata module для `_market_data`;
- налаштувати Alembic environment для market data migrations;
- створити `_market_data`;
- додати порожні таблиці `market_symbols`, `provider_symbols`;
- додати Alembic migration tests або schema inspection smoke;
- не переносити candles і не перемикати readers.

Готовність:

- Alembic migration застосовується й відкочується за стандартною процедурою;
- старі таблиці `_candles_trading_data` не змінені.

### Етап 2. Symbol registry seed

Мета: централізувати mapping canonical/provider symbols.

- заповнити `market_symbols`;
- заповнити `provider_symbols` для Binance Spot;
- підтримати `1h`, `4h`, `1d`;
- додати validation, що кожен configured trading symbol має provider mapping;
- заблокувати sync для symbols без active mapping.

Готовність:

- registry покриває initial symbol universe;
- hardcoded mapping не додається в новий data service.

### Етап 3. Canonical candle tables per timeframe

Мета: додати canonical storage для свічок без batch/snapshot логіки.

- описати логічну `market_candles` model у SQLAlchemy Core metadata;
- створити parent `market_candles` і physical timeframe tables або partitions:
  - `market_candles_1h`;
  - `market_candles_4h`;
  - `market_candles_1d`;
- у кожній timeframe table зберігати всі символи через колонку `canonical_symbol`;
- додати unique key `source + canonical_symbol + timeframe + open_time`;
- реалізувати idempotent insert через `on conflict(source, canonical_symbol, timeframe, open_time) do nothing`;
- реалізувати pure normalizer для OHLCV;
- реалізувати closed-candle validator;
- додати unit tests для interval duration, ordering, closed candle і Decimal/NUMERIC coercion.

Готовність:

- можна безпечно повторно вставляти candles без помилки;
- додавання нового symbol не потребує migration;
- duplicate candles блокуються unique constraint на рівні БД і пропускаються через `do nothing`;
- існуючі застосунки ще не читають нову таблицю.

### Етап 4. Binance read-only provider adapter

Мета: винести завантаження candles у один provider adapter.

- реалізувати `fetch_closed_candles`;
- використовувати symbol registry;
- додати server-time або local closed-candle boundary validation;
- додати retryable/non-retryable error classification;
- не писати в БД напряму з adapter-а.

Готовність:

- adapter повертає canonical candles;
- adapter не знає про стратегії;
- adapter не має доступу до private exchange credentials.

### Етап 5. Single-symbol sync command

Мета: навчитися синхронізувати один `symbol + timeframe` у canonical candle storage.

- створити sync use case для одного `source + symbol + timeframe + range`;
- додати PostgreSQL advisory lock на `source + symbol + timeframe` через repository;
- писати candles у відповідну timeframe table/partition, де всі symbols зберігаються разом;
- заборонити будь-яку побудову candle table name із symbol;
- реалізувати idempotent bulk insert через SQLAlchemy Core PostgreSQL insert/on conflict do nothing або контрольований raw SQL;
- не створювати snapshots і events;
- додати dry-run/log-only режим для перевірки provider response.

Готовність:

- повторний sync не створює дублікати;
- parallel duplicate sync блокується або завершується no-op;
- існуючі sync paths усе ще працюють незалежно.

### Етап 6. Gap detection

Мета: не дозволяти неповні дані як input для майбутніх snapshots.

- реалізувати interval gap detector;
- перевіряти duplicate intervals, missing intervals і wrong duration;
- додати status result `COMPLETE | INCOMPLETE | GAP_DETECTED | STALE`;
- додати тести на gaps у середині lookback і на missing last candle.

Готовність:

- sync може відрізнити повний range від неповного;
- ready snapshot events ще не публікуються.

### Етап 7. Batch tracking

Мета: зробити sync audit-able.

- описати `market_data_batches` у SQLAlchemy Core metadata;
- створити `market_data_batches` через Alembic migration;
- кожен sync створює batch `RUNNING`;
- завершувати batch у `COMPLETE`, `INCOMPLETE` або `FAILED`;
- записувати rows fetched/inserted/skipped_duplicate/hash_mismatch, gap count, error code;
- додати `outbox_status`, але ще не публікувати events.

Готовність:

- кожен sync має batch record;
- failures не губляться в логах.

### Етап 8. Snapshot model

Мета: створити immutable market data input для майбутніх consumers.

- описати `market_snapshots` і `market_snapshot_candles` у SQLAlchemy Core metadata;
- створити `market_snapshots` і `market_snapshot_candles` через Alembic migration;
- рахувати `data_hash`;
- додати `snapshot_version`;
- snapshot reader має читати candles тільки через `market_snapshot_candles`;
- при незмінному `data_hash` не створювати новий logical snapshot.

Готовність:

- snapshot можна відтворити після audited candle correction/backfill;
- audit може перевірити candle hashes.

### Етап 9. Complete batch -> snapshot transaction

Мета: атомарно завершувати sync даних і snapshot.

- в одній SQLAlchemy async transaction вставляти candles через `ON CONFLICT DO NOTHING`, завершувати batch і створювати snapshot;
- якщо транзакція падає, не має лишатися partial complete state;
- retry має знайти existing complete snapshot, якщо `data_hash` не змінився;
- новий `snapshot_version` створюється тільки якщо audited correction/backfill змінив `data_hash` або completeness status;
- downstream consumers ще не запускаються.

Готовність:

- `COMPLETE` batch завжди має валідний snapshot;
- `INCOMPLETE/FAILED` batch не має ready snapshot.

### Етап 10. Transactional outbox для `CandleBatchReady`

Мета: публікувати готовність candle snapshot після commit.

- описати `outbox_events` у SQLAlchemy Core metadata;
- створити `outbox_events` через Alembic migration;
- записувати `CandleBatchReady` в тій самій транзакції, що й complete batch/snapshot;
- використовувати idempotency key із `snapshot_version`;
- не доставляти подію напряму з sync transaction.

Готовність:

- complete sync створює pending outbox event;
- duplicate sync не створює duplicate logical event.

### Етап 11. Outbox publisher

Мета: доставляти pending events у queue/broker і переживати restart.

- реалізувати publisher worker;
- читати й оновлювати outbox rows через SQLAlchemy Core repository;
- додати retry/backoff;
- відмічати `PUBLISHED` тільки після підтвердження broker-а;
- додати dead-letter або `FAILED` після max attempts;
- додати publisher lag metric.

Готовність:

- restart publisher-а не губить pending events;
- повторна доставка допустима, consumers мають бути idempotent.

### Етап 12. Market data scheduler

Мета: замінити ручний sync у межах самого Market Data Service на централізований scheduler.

- планувати `1h`, `4h`, `1d` sync jobs за UTC close boundaries;
- додати safety delay;
- додати jitter для багатьох symbols;
- не змінювати існуючі application sync paths поза `market-data-service`.

Готовність:

- scheduler сам створює sync jobs;
- per-symbol events приходять поступово після готовності даних.

### Етап 13. Backfill jobs

Мета: автоматично відновлювати gaps.

- створювати `MarketDataBackfillRequested` при gaps;
- backfill має нижчий priority, ніж fresh candle sync;
- після backfill створювати новий snapshot version тільки якщо змінився `data_hash` або completeness status;
- не публікувати ready snapshot event для incomplete range.

Готовність:

- gap автоматично переводиться в backfill flow;
- після backfill complete snapshot може бути створений без ручної дії.

### Етап 14. Observability baseline

Мета: зробити data layer операційно видимим.

- додати metrics із секції Observability;
- додати dashboards;
- додати alerts на stale snapshots, gaps, provider failures, outbox lag;
- додати structured logs із batch/snapshot/correlation ids.

Готовність:

- оператор бачить, чому snapshot/event не створився;
- stale/gap/provider failure не лишаються silent.

### Етап 15. Failure та recovery tests

Мета: довести production reliability.

- provider timeout test;
- partial response test;
- duplicate job test;
- SQLAlchemy async transaction rollback test;
- outbox publisher restart test;
- snapshot immutability test після candle correction.

Готовність:

- усі documented failure modes мають automated або scripted smoke coverage.

### Етап 16. Load і rate-limit hardening

Мета: перевірити сервіс на production scale.

- load test на expected symbols/timeframes;
- перевірити Binance rate limits;
- налаштувати concurrency limits;
- додати circuit breaker behavior;
- перевірити queue lag і outbox lag під навантаженням.

Готовність:

- сервіс витримує очікуваний symbol universe із запасом;
- rate-limit errors не валять весь sync cycle.

### Етап 17. Production rollout

Мета: безпечно запустити Market Data Service як production data-ingestion layer без підключення ботів.

- rollout по одному symbol group або timeframe group;
- перевірити alerts;
- зафіксувати rollback decision points;
- після стабільного періоду зафіксувати сервіс як готовий до окремого плану інтеграції consumers.

Готовність:

- Market Data Service стабільно створює candles, batches, snapshots і outbox events;
- rollback procedure перевірена до завершення rollout.

## 19. Критерії Готовності

- Новий symbol додається через registry seed/backfill, а не через schema migration.
- Один `symbol + timeframe + close_time` sync не створює дублікати.
- Complete sync створює відтворюваний snapshot і `CandleBatchReady`.
- Incomplete, stale або gap data не отримує ready snapshot event.
- Stale policy формально визначена для кожного timeframe.
- Snapshot є відтворюваним через `market_snapshot_candles`, навіть якщо пізніше виконано audited candle correction/backfill.
- Symbol mapping береться тільки з registry, а не з hardcoded adapter logic.
- Події створюються тільки після DB commit.
- Повторна доставка подій не створює duplicate logical outbox events.
- Restart workers не губить batches, snapshots або pending events.
- Є metrics, alerts і audit trail для кожного batch/snapshot.
- Rollback path задокументований і перевірений.
