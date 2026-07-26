# План допрацювання Python-сервісів до постійних backend-сервісів

Документ описує технічний план переведення поточних Python-сервісів з набору доменних/application компонентів, міграцій і Docker jobs у повноцінні long-running backend-сервіси.

Пріоритет виконання: спочатку `market-data-service`, потім `bot-platform-service`, після цього інтеграція з `control-panel`.

Виняток: read-only інтеграцію `control-panel/api` для `GET /admin/bot-modules` і `GET /admin/bot-modules/{moduleId}/config-schema` можна зробити раніше як тимчасовий міст, бо потрібні дані вже лежать у `_bot_platform.bot_modules` після `bot_modules_sync`. Цей міст не повинен містити lifecycle логіку bot instances/runs.

## 1. Market Data Service

### 1.1. Цільовий стан

`market-data-service` має стати постійним backend-сервісом, який:

- запускається як long-running process у Docker Compose;
- має production CLI entrypoint;
- самостійно збирає runtime object graph / application dependencies через composition root;
- запускає scheduler loop для збору market data;
- запускає outbox publisher loop для Redis Stream / downstream events;
- має health/readiness endpoint;
- має metrics endpoint;
- коректно завершується по `SIGTERM` / `SIGINT`;
- підтримує one-shot CLI jobs для міграцій, backfill, smoke-перевірок і maintenance;
- не залежить від legacy bot services.

### 1.2. Поточна база

У сервісі вже є значна частина внутрішньої логіки:

- доменні моделі market symbols, candles, snapshots, sync jobs, batches, outbox events;
- SQLAlchemy Core таблиці та repositories;
- Alembic migrations;
- Binance Spot adapter;
- Redis Stream broker;
- application services для symbol registry, sync, scheduler, outbox publisher, backfill planning;
- worker classes:
  - `MarketDataSchedulerWorker`;
  - `OutboxPublisherWorker`;
- observability modules для metrics, alerts, structured logging;
- Docker image і Docker Compose migration job.

Ключова прогалина: немає production composition root, який збирає ці частини в один керований long-running процес.

### 1.3. Архітектура runtime-процесу

Потрібно додати application entrypoint з runtime командами:

```text
python -m market_data_service.main serve
python -m market_data_service.main scheduler
python -m market_data_service.main outbox-publisher
python -m market_data_service.main healthcheck
```

Окремо потрібні one-shot job / maintenance команди:

```text
python -m market_data_service.main db:check
python -m market_data_service.main db:revision
python -m market_data_service.main backfill
python -m market_data_service.main symbols:sync
python -m market_data_service.main gaps:scan
python -m market_data_service.main outbox:replay
```

Рекомендований production режим:

```text
serve
```

У режимі `serve` процес має підняти:

- lifecycle manager;
- DB engine / connection pool;
- Redis client;
- provider client;
- scheduler worker task;
- outbox publisher worker task;
- HTTP server для health/readiness/metrics;
- graceful shutdown coordinator.

Окремі режими `scheduler` і `outbox-publisher` варто залишити для майбутнього горизонтального масштабування, коли ці loops потрібно буде запускати окремими deployment units.

### 1.4. Composition Root

Створити явний composition layer, який відповідає тільки за wiring:

```text
market_data_service/runtime/
  app.py
  container.py
  lifecycle.py
  settings.py
  signals.py
```

Відповідальність:

- прочитати env config;
- провалідувати required settings;
- створити SQLAlchemy async engine;
- створити Redis client;
- створити HTTP provider adapter;
- зібрати repositories;
- зібрати application services;
- зібрати worker-и;
- зареєструвати shutdown hooks;
- віддати готовий runtime app.

У composition root не повинно бути бізнес-логіки market data.

### 1.5. HTTP Health API

Додати легкий HTTP server. Достатньо FastAPI або aiohttp. FastAPI зручніший для майбутніх admin/debug endpoint-ів, але треба додати залежності:

- `fastapi`;
- `uvicorn`;
- опційно `prometheus-client`.

Мінімальні endpoint-и:

```text
GET /health/live
GET /health/ready
GET /metrics
```

`/health/live`:

- повертає 200, якщо event loop живий і процес не в shutdown.

`/health/ready`:

- перевіряє PostgreSQL `select 1`;
- перевіряє Redis `PING`;
- перевіряє, що migrations сумісні з runtime revision;
- перевіряє scheduler/outbox loop health тільки якщо відповідний loop увімкнений у конфігурації;
- не має ходити в Binance API, щоб readiness не залежав від зовнішнього provider-а.

На ранньому етапі, коли HTTP server уже є, але scheduler/outbox loops ще не підключені, readiness має перевіряти тільки PostgreSQL, Redis і migration compatibility. Після підключення scheduler/outbox readiness розширюється умовними checks для enabled loops.

`/metrics`:

- експортує counters/gauges/histograms для Prometheus.

Docker healthcheck має використовувати один canonical readiness contract. Технічно правильний варіант:

- HTTP endpoints є source of truth для liveness/readiness;
- CLI `healthcheck` не дублює health логіку, а робить локальний HTTP запит до `http://127.0.0.1:${MARKET_DATA_HTTP_PORT}/health/ready`;
- якщо HTTP server не стартував або readiness повертає non-2xx, CLI завершується з non-zero exit code.

Це прибирає ризик двох різних health implementation: одна для HTTP endpoint-ів і друга для Docker healthcheck.

### 1.6. Worker Lifecycle

Для кожного loop потрібен єдиний контракт:

```text
start()
stop()
run_forever()
health()
```

Поточні `MarketDataSchedulerWorker` і `OutboxPublisherWorker` уже мають `run_once()` і `run_forever()`, але ще не мають повного lifecycle contract. У межах цього етапу їх треба розширити або обгорнути lifecycle adapter-ом, щоб runtime process міг керовано стартувати, зупиняти і health-check-ити кожен loop.

Scheduler worker:

- читає configured symbols/timeframes;
- планує next due windows;
- бере advisory lock або інший DB lock для уникнення duplicate ingestion;
- викликає sync service;
- пише batch/snapshot/outbox records;
- має backoff після provider errors;
- не падає весь процес через одну помилку symbol/timeframe.

Outbox publisher:

- читає unpublished outbox events;
- публікує у Redis Stream;
- маркує event як published;
- підтримує retry/backoff;
- має idempotency на рівні event id;
- не блокує scheduler loop.

### 1.7. Graceful Shutdown

На `SIGTERM` / `SIGINT`:

- перестати приймати нові HTTP запити, крім health shutdown state;
- виставити process state `shutting_down`;
- зупинити scheduler planning нових задач;
- дочекатися завершення поточної sync операції або timeout;
- зупинити outbox publisher після поточного publish batch;
- закрити Redis client;
- закрити DB engine;
- завершити процес з exit code `0`.

Додати конфіг:

```text
MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS=30
```

### 1.8. Конфігурація

Звести всі env vars у типізований settings layer.

Групи конфігів:

- database;
- redis/outbox;
- provider;
- scheduler;
- backfill;
- HTTP server;
- logging;
- metrics;
- shutdown;
- feature flags.

Обов'язкові змінні мають fail-fast перевірку при старті.

Рекомендовані нові env vars:

```text
MARKET_DATA_SERVICE_MODE=serve
MARKET_DATA_HTTP_HOST=0.0.0.0
MARKET_DATA_HTTP_PORT=8010
MARKET_DATA_LOG_LEVEL=INFO
MARKET_DATA_METRICS_ENABLED=true
MARKET_DATA_SCHEDULER_ENABLED=true
MARKET_DATA_OUTBOX_PUBLISHER_ENABLED=true
MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS=30
```

### 1.9. Database Migration Strategy

Міграції не повинні автоматично виконуватися всередині long-running service container при кожному старті.

Залишити окремий Docker job:

```text
market_data_migrate
```

Long-running service має:

- перевіряти, що schema revision сумісна;
- падати з явною помилкою, якщо migrations не застосовані;
- не створювати таблиці самостійно.

CLI режим `migrate` у long-running entrypoint не є обов'язковим і не має бути основним production шляхом. Основний production шлях - окремий Docker job `market_data_migrate`, який явно викликає Alembic. Якщо CLI `db:migrate` буде додано для локальної зручності, він має бути wrapper-ом над Alembic і не запускатися автоматично у `serve`.

Додати CLI:

```text
python -m market_data_service.main db:revision
python -m market_data_service.main db:check
```

### 1.10. Backfill і Maintenance Jobs

Окремі one-shot jobs:

```text
python -m market_data_service.main backfill --symbol BTCUSDT --timeframe 1h --from ... --to ...
python -m market_data_service.main symbols:sync
python -m market_data_service.main gaps:scan
python -m market_data_service.main outbox:replay --from-id ...
```

Ці команди мають використовувати той самий composition root, але без запуску HTTP server і worker loops.

### 1.11. Docker Compose

Додати long-running service:

```text
market_data:
  build:
    context: ../../market-data-service
  command: python -m market_data_service.main serve
  depends_on:
    market_data_migrate:
      condition: service_completed_successfully
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
  healthcheck:
    test: ["CMD", "python", "-m", "market_data_service.main", "healthcheck"]
```

`healthcheck` має перевіряти HTTP readiness endpoint поточного процесу, а не напряму ходити в PostgreSQL/Redis окремою логікою.

Залишити окремо:

- `market_data_migrate`;
- майбутній `market_data_backfill`;
- майбутній `market_data_symbols_sync`;
- майбутній `market_data_gaps_scan`.

### 1.12. Observability

Structured logs:

- `service`;
- `environment`;
- `correlation_id`;
- `symbol`;
- `timeframe`;
- `provider`;
- `batch_id`;
- `sync_job_id`;
- `event_id`;
- `duration_ms`;
- `error_code`.

Metrics:

- scheduler loop iterations;
- sync success/failure count;
- provider request latency;
- provider rate-limit errors;
- candles ingested;
- snapshots written;
- outbox events pending/published/failed;
- Redis publish latency;
- DB query latency for hot paths;
- worker liveness timestamp.

Alerts:

- scheduler loop stale;
- outbox pending backlog above threshold;
- provider error rate above threshold;
- no candles ingested for active symbol/timeframe;
- DB unavailable;
- Redis unavailable.

### 1.13. Reliability

Потрібно передбачити:

- idempotent candle writes;
- idempotent outbox publish;
- advisory locks для scheduler/backfill;
- retry з exponential backoff;
- circuit breaker для Binance;
- concurrency limit per provider;
- timeout-и на всі network calls;
- bounded task queue;
- isolation між symbol/timeframe jobs;
- crash recovery через persisted sync jobs/outbox.

### 1.14. Security

Для market data сервісу private exchange keys не потрібні.

Мінімальні вимоги:

- не логувати повні database URLs;
- не логувати Redis credentials;
- не відкривати debug endpoints назовні;
- metrics endpoint у production бажано тримати тільки в internal network;
- HTTP admin/maintenance endpoints не додавати без auth.

### 1.15. Тести

Додати рівні тестів:

- unit tests для settings validation;
- unit tests для lifecycle manager;
- unit tests для graceful shutdown;
- unit tests для health readiness aggregation;
- integration tests з PostgreSQL і Redis;
- smoke test Docker Compose: migrations + service health + one scheduler iteration;
- failure tests: DB down, Redis down, provider timeout, duplicate scheduler instance.

Мінімальний acceptance набір:

```text
pytest tests/unit
pytest tests/integration
docker compose ... up --build market_data_migrate market_data
curl http://localhost:8010/health/ready
curl http://localhost:8010/metrics
```

### 1.16. Acceptance Criteria для Market Data Service

Market Data Service можна вважати повноцінним backend-сервісом, коли:

- container `market_data` залишається running після старту;
- healthcheck переходить у healthy;
- readiness падає, якщо PostgreSQL або Redis недоступні;
- scheduler loop реально пише market data у БД;
- outbox publisher публікує events у Redis Stream;
- процес коректно завершується по `docker stop`;
- logs і metrics дозволяють зрозуміти стан сервісу без shell-доступу;
- повторний запуск не створює дублікати даних;
- migration job і long-running service розділені.

## 2. Bot Platform Service

### 2.1. Цільовий стан

`bot-platform-service` має стати постійним backend-сервісом, який:

- читає registered bot modules;
- управляє bot instances;
- валідовує bot configs;
- запускає signal-only bot runs;
- читає snapshots через стабільний `market-data-service` contract;
- пише bot runs, audit events, state і signals;
- публікує standardized trading signals;
- має health/readiness/metrics;
- не імпортує legacy bot services.

### 2.2. Залежність від Market Data Service

Цей сервіс варто допрацьовувати після `market-data-service`, бо bot runs потребують стабільного джерела market snapshots.

Перед стартом повноцінного bot runtime потрібні:

- стабільна schema market snapshots;
- contract для читання snapshots;
- гарантія freshness market data;
- health/readiness сигнал від market data layer.

Цільовий технічно правильний boundary: `bot-platform-service` не повинен напряму залежати від внутрішніх деталей реалізації `market-data-service`. Для довгострокового контракту потрібен один із двох явних варіантів:

1. HTTP/read API у `market-data-service` для latest complete snapshots.
2. Versioned read model у PostgreSQL, офіційно задокументований як public storage contract між сервісами.

Поточний direct DB read provider у `bot-platform-service` можна залишити як перехідний варіант, але його треба оформити як залежність від versioned read model, а не як довільне читання internal tables. Для нового runtime краще закласти HTTP/read API або формально стабілізований read model.

### 2.3. Runtime режими

CLI режими:

```text
python -m bot_platform_service.main serve
python -m bot_platform_service.main runner
python -m bot_platform_service.main bot:modules:sync
python -m bot_platform_service.main db:check
python -m bot_platform_service.main healthcheck
```

`serve`:

- HTTP API;
- optional runner loop;
- optional signal event publisher, disabled until signal publishing stage;
- health/metrics.

`runner`:

- тільки bot execution loop для масштабування окремо від API.

Рекомендовано розділити API і runner operationally:

- `serve` піднімає admin/runtime HTTP API і може мати runner disabled через env flag;
- `runner` запускає тільки execution loop;
- у Docker Compose можна стартувати один контейнер `bot_platform_api` і окремий контейнер `bot_platform_runner`, коли з'явиться потреба масштабування.

`bot:modules:sync`:

- залишається one-shot Docker job.

### 2.4. HTTP API

Мінімальні endpoint-и:

```text
GET /health/live
GET /health/ready
GET /metrics
GET /admin/bot-modules
GET /admin/bot-modules/{module_id}/config-schema
POST /admin/bot-instances/validate-config
POST /admin/bot-instances
GET /admin/bot-instances
POST /admin/bot-instances/{instance_id}/enable
POST /admin/bot-instances/{instance_id}/pause
POST /admin/bot-instances/{instance_id}/run
```

Спочатку можна зробити internal API без public exposure, а `control-panel/api` буде facade/proxy.

### 2.5. Worker Lifecycle

Runner loop:

- знаходить enabled bot instances;
- перевіряє freshness market snapshots;
- бере advisory lock на instance/run;
- створює `bot_runs`;
- викликає platform-native adapter;
- валідовує output signal contract;
- пише `bot_signals`;
- пише audit events;
- після етапу signal event publisher передає persisted signals у downstream event transport.

### 2.6. Docker Compose

Базові one-shot jobs:

```text
bot_platform_migrate
bot_modules_sync
```

Цільові long-running services:

```text
bot_platform_api
bot_platform_runner
```

`bot_platform_api` має залежати від:

- `bot_platform_migrate`;
- `bot_modules_sync`;
- `postgres`;
- `redis`, якщо API використовує Redis-backed readiness, cache або event transport.

`bot_platform_runner` має залежати від:

- `bot_platform_api`;
- `market_data`;
- `postgres`;
- `redis`.

Залежність від `market_data` має бути не для всього Bot Platform API, а для runner/runtime execution. Admin metadata API, який показує modules/config schemas у control-panel, повинен працювати після `bot_platform_migrate` і `bot_modules_sync`, навіть якщо market-data ingestion ще не healthy.

Рекомендований поділ:

```text
bot_platform_api:
  depends_on:
    bot_platform_migrate: service_completed_successfully
    bot_modules_sync: service_completed_successfully
    postgres: service_healthy

bot_platform_runner:
  depends_on:
    bot_platform_api: service_healthy
    market_data: service_healthy
    postgres: service_healthy
    redis: service_healthy
```

На першому етапі можна тимчасово залишити один контейнер `bot_platform` як API shell, але в плані реалізації треба не блокувати read-only admin API залежністю від `market_data`. Коли з'являється runner, назви і відповідальності треба розділити на `bot_platform_api` і `bot_platform_runner`.

### 2.7. Acceptance Criteria для Bot Platform Service

- container `bot_platform_api` залишається running;
- container `bot_platform_runner` залишається running після етапу runner;
- modules sync працює як one-shot job;
- `/admin/bot-modules` повертає `spot_grid` і `spot_greenwich`;
- config schema доступна для кожного module;
- можна створити bot instance;
- можна запустити signal-only dry run;
- run створює `bot_runs`, `bot_run_events`, `bot_signals`;
- service не відкриває exchange orders;
- legacy bot directories не потрібні для runtime.

## 3. Інтеграція з Control Panel

### 3.1. Поточний стан

Frontend вже має сторінку bot configuration і очікує маршрути:

```text
GET /admin/bot-modules
GET /admin/bot-modules/{moduleId}/config-schema
POST /admin/bot-instances/validate-config
```

Але `control-panel/api` ще має реалізувати backend частину цих routes або проксі до `bot-platform-service`.

### 3.2. Варіанти інтеграції

Варіант A: `control-panel/api` читає `_bot_platform` таблиці напряму.

Плюси:

- швидше реалізувати;
- менше network hops;
- простіше для першого admin UI.

Мінуси:

- control-panel отримує знання про internal schema bot platform;
- складніше змінювати bot-platform schema.

Варіант B: `control-panel/api` викликає HTTP API `bot-platform-service`.

Плюси:

- чистіша service boundary;
- bot-platform володіє своїми правилами;
- легше розвивати lifecycle операції.

Мінуси:

- треба спочатку підняти HTTP API bot-platform-service;
- більше конфігурації і failure modes.

Рекомендація:

- короткостроково: варіант A тільки для read-only module list/schema, щоб control-panel міг показати зареєстровані модулі;
- середньостроково: варіант B для config validation, instance lifecycle і run operations;
- довгостроково: `bot-platform-service` володіє всіма bot-domain правилами, а `control-panel/api` залишається auth/facade layer.

## 4. Рекомендована послідовність робіт

Принцип планування: кожен етап має давати малий перевірний результат, не ламати попередні Docker jobs і не вимагати одночасної реалізації кількох незалежних runtime loops. Якщо етап не можна перевірити окремо, його треба розбити ще дрібніше.

### Етап 1. Market Data CLI skeleton

Мета: додати контрольований entrypoint без запуску long-running логіки.

1. Додати `market_data_service.main`.
2. Додати command routing для `serve`, `scheduler`, `outbox-publisher`, `backfill`, `healthcheck`, `db:check`, `db:revision`.
3. Для ще не реалізованих команд повертати явний unsupported/not implemented exit code.
4. Не змінювати існуючий `market_data_migrate` Docker job.

Acceptance criteria:

- `python -m market_data_service.main --help` працює;
- `python -m market_data_service.main db:check` може перевірити доступність БД або повернути зрозумілу помилку конфігурації;
- migration job продовжує запускати Alembic напряму.

### Етап 2. Market Data settings layer

Мета: централізувати конфігурацію без зміни поведінки worker-ів.

1. Додати typed settings для database, Redis, provider, scheduler, HTTP, logging, shutdown.
2. Зберегти сумісність з поточними env vars `DB_*`, `MARKET_DATA_DATABASE_URL`, `MARKET_DATA_REDIS_URL`.
3. Додати fail-fast validation для required settings.
4. Додати unit tests для default values і помилкових env комбінацій.

Acceptance criteria:

- settings можна створити локально і в Docker env;
- помилки конфігурації мають зрозумілий текст;
- поточні migration jobs не потребують зміни env names.

### Етап 3. Market Data composition root

Мета: зібрати runtime object graph без запуску HTTP server або infinite loops.

1. Додати runtime container/composition module.
2. Створювати SQLAlchemy async engine.
3. Створювати Redis client.
4. Створювати provider adapter.
5. Збирати repositories, application services і worker instances.
6. Додати cleanup/dispose метод.

Acceptance criteria:

- composition root стартує і закривається у unit/integration test;
- DB/Redis clients закриваються без resource warnings;
- бізнес-логіка не переноситься у composition layer.

### Етап 4. Market Data HTTP liveness/readiness

Мета: підняти мінімальний HTTP server без scheduler/outbox loops.

1. Додати HTTP server.
2. Додати `GET /health/live`.
3. Додати `GET /health/ready`.
4. Додати `GET /metrics` як stub або базовий Prometheus endpoint.
5. Додати CLI `healthcheck`, який ходить у локальний `/health/ready`.

Acceptance criteria:

- `serve` може стартувати HTTP server;
- `/health/live` повертає 200;
- `/health/ready` перевіряє PostgreSQL і Redis;
- Docker healthcheck використовує той самий readiness contract.

### Етап 5. Market Data lifecycle і graceful shutdown

Мета: зробити керований runtime process до підключення бізнес-loop-ів.

1. Додати lifecycle manager.
2. Додати process state: starting, ready, shutting_down.
3. Обробити `SIGTERM` і `SIGINT`.
4. Закривати HTTP server, Redis client і DB engine.
5. Додати shutdown timeout.

Acceptance criteria:

- `docker stop` завершує process без stack trace;
- readiness повертає non-2xx під час shutdown;
- shutdown не залишає відкриті DB/Redis connections.

### Етап 6. Market Data scheduler lifecycle adapter

Мета: підключити scheduler окремо від outbox publisher.

1. Розширити або обгорнути `MarketDataSchedulerWorker` lifecycle contract-ом.
2. Додати `MARKET_DATA_SCHEDULER_ENABLED`.
3. Підключити scheduler task у `serve`.
4. Додати ізоляцію помилок одного tick від падіння process.
5. Додати базові metrics/logs для scheduler loop.

Acceptance criteria:

- scheduler можна вмикати/вимикати env flag-ом;
- один failed tick не завершує container;
- при shutdown scheduler не планує нові задачі.

### Етап 7. Market Data outbox publisher lifecycle adapter

Мета: підключити outbox publisher після стабільного scheduler lifecycle.

1. Розширити або обгорнути `OutboxPublisherWorker` lifecycle contract-ом.
2. Додати `MARKET_DATA_OUTBOX_PUBLISHER_ENABLED`.
3. Підключити publisher task у `serve`.
4. Додати retry/backoff для publish failures.
5. Додати базові metrics/logs для pending/published/failed events.

Acceptance criteria:

- publisher можна вмикати/вимикати env flag-ом;
- Redis failure не завершує весь process без контрольованої помилки;
- shutdown завершує поточний publish batch або timeout.

### Етап 8. Market Data Docker service

Мета: додати long-running Docker service без зміни migration job.

1. Додати service `market_data`.
2. Залежати від `market_data_migrate`, `postgres`, `redis`.
3. Додати healthcheck через CLI `healthcheck`.
4. Прокинути HTTP port тільки для local/dev, якщо потрібна ручна перевірка.
5. Не запускати migrations у `market_data` service.

Acceptance criteria:

- `market_data_migrate` завершується успішно;
- `market_data` залишається running;
- container переходить у healthy;
- повторний `docker compose up` не дублює schema/data.

### Етап 9. Market Data minimal smoke scenario

Мета: перевірити реальний runtime path end-to-end.

1. Підняти PostgreSQL, Redis, `market_data_migrate`, `market_data`.
2. Перевірити `/health/ready`.
3. Запустити один контрольований scheduler tick або короткий scheduler interval.
4. Перевірити запис snapshot/candle/batch у БД.
5. Перевірити publish event у Redis Stream.

Acceptance criteria:

- є автоматизований smoke test або documented smoke command;
- smoke не залежить від legacy bot directories;
- помилка provider-а має зрозумілий лог і exit/test result.

### Етап 10. Market Data maintenance command: symbols sync

Мета: додати перший one-shot maintenance job окремо від long-running service.

1. Додати CLI `symbols:sync`.
2. Використати існуючий composition root без HTTP server.
3. Додати Docker job `market_data_symbols_sync`.
4. Додати тест idempotency.

Acceptance criteria:

- повторний запуск не створює дублікати symbols;
- job завершується з exit code 0 при успіху;
- помилки конфігурації повертають non-zero exit code.

### Етап 11. Market Data maintenance command: backfill

Мета: додати backfill як окремий керований job.

1. Додати CLI `backfill`.
2. Додати аргументи symbol/timeframe/from/to.
3. Використати advisory lock або persisted job state.
4. Додати Docker job `market_data_backfill`.
5. Додати тест на повторний запуск.

Acceptance criteria:

- backfill не конфліктує з scheduler;
- повторний запуск idempotent;
- великі діапазони можна обмежити batch size/concurrency config-ом.

### Етап 12. Market Data maintenance command: gaps scan

Мета: додати діагностику gaps після базового ingest/backfill.

1. Додати CLI `gaps:scan`.
2. Перевіряти configured symbols/timeframes або явно передані аргументи.
3. Писати зрозумілий report.
4. Опційно створювати backfill requests.

Acceptance criteria:

- команда може працювати read-only;
- report придатний для ручного troubleshooting;
- команда не запускає backfill без явного прапора.

### Етап 13. Market Data maintenance command: outbox replay

Мета: додати recovery path для outbox після Redis або publisher failure.

1. Додати CLI `outbox:replay`.
2. Додати аргументи `--from-id`, `--to-id`, `--dry-run`.
3. Забезпечити idempotent publish.
4. Додати тест duplicate-safe replay.

Acceptance criteria:

- dry-run не змінює стан;
- replay не створює неконтрольовані дублікати;
- помилки Redis мають non-zero exit code.

### Етап 14. Market Data formal snapshot contract

Мета: підготувати стабільний контракт для `bot-platform-service`.

1. Вибрати контракт: HTTP/read API або versioned PostgreSQL read model.
2. Задокументувати schema/response contract.
3. Додати compatibility/version field.
4. Додати tests для latest complete snapshot і stale/not ready cases.

Acceptance criteria:

- `bot-platform-service` може читати snapshots без знання internal implementation details;
- contract має versioning;
- breaking changes мають migration path.

### Етап 15. Bot Platform settings і composition root

Мета: підготувати runtime основу без HTTP API і runner loop.

1. Додати typed settings.
2. Додати composition root.
3. Збирати DB engine, repositories, admin metadata service.
4. Зберегти існуючий `bot:modules:sync`.

Acceptance criteria:

- `bot:modules:sync` працює без регресій;
- composition root стартує і закривається в тесті;
- legacy bot directories не потрібні для runtime.

### Етап 16. Bot Platform HTTP health/metrics

Мета: підняти API shell без bot lifecycle операцій.

1. Додати HTTP server.
2. Додати `/health/live`.
3. Додати `/health/ready`.
4. Додати `/metrics`.
5. Додати CLI `healthcheck`, який перевіряє HTTP readiness.

Acceptance criteria:

- service стартує як long-running process;
- readiness перевіряє PostgreSQL;
- Docker healthcheck не дублює health логіку.

### Етап 17. Bot Platform admin metadata API

Мета: відкрити read-only API для modules/config schemas.

1. Додати `GET /admin/bot-modules`.
2. Додати `GET /admin/bot-modules/{module_id}/config-schema`.
3. Використати існуючий `AdminMetadataService`.
4. Не імпортувати strategy code під час read.

Acceptance criteria:

- API повертає `spot_grid` і `spot_greenwich` після `bot_modules_sync`;
- schema доступна для кожного module;
- API не залежить від `market_data`.

### Етап 18. Control Panel read-only bot modules

Мета: показати bot modules у control-panel без очікування повного bot runner.

1. Додати `control-panel/api` routes для module list/schema.
2. Якщо `bot-platform-service` HTTP API ще не готовий, тимчасово читати `_bot_platform.bot_modules` напряму.
3. Обмежити direct DB read тільки read-only metadata.
4. Підключити існуючий frontend module-admin-bots.

Acceptance criteria:

- control-panel показує dropdown з `spot_grid` і `spot_greenwich`;
- config schema завантажується;
- жодних create/run lifecycle операцій через direct DB.

### Етап 19. Bot Platform config validation API

Мета: перенести validation правил у bot-platform boundary.

1. Додати `POST /admin/bot-instances/validate-config`.
2. Валідовувати config проти persisted config schema.
3. Повернути стабільний error format.
4. Підключити control-panel через `control-panel/api` facade/proxy.

Acceptance criteria:

- валідна config проходить validation;
- невалідна config повертає field-level errors;
- control-panel більше не потребує direct validation логіки.

### Етап 20. Bot Platform instance lifecycle API

Мета: додати CRUD/lifecycle без runner execution.

1. Додати create/list bot instances.
2. Додати enable/pause/disable.
3. Писати audit events.
4. Не запускати bot runs на цьому етапі.

Acceptance criteria:

- instance можна створити і побачити у списку;
- status transitions валідовані;
- audit events пишуться.

### Етап 21. Bot Platform runner skeleton

Мета: додати runner process без реального signal generation.

1. Додати CLI `runner`.
2. Додати Docker service `bot_platform_runner`.
3. Runner читає enabled instances.
4. Runner перевіряє market data readiness/snapshot availability.
5. Runner не викликає adapters, якщо snapshot не ready.

Acceptance criteria:

- runner залишається running;
- runner не блокує admin API;
- market-data unavailable не ламає admin API.

### Етап 22. Bot Platform single manual run

Мета: додати контрольований запуск одного bot instance.

1. Додати `POST /admin/bot-instances/{instance_id}/run`.
2. Брати lock на instance/run.
3. Створювати `bot_runs`.
4. Читати latest complete snapshot через формальний contract.
5. Викликати platform-native adapter.
6. Писати run result без signal publisher fanout.

Acceptance criteria:

- manual run створює `bot_runs`;
- duplicate run lock працює;
- stale snapshot повертає контрольовану помилку.

### Етап 23. Bot Platform signal persistence

Мета: зберігати standardized signals після run.

1. Валідовувати output signal contract.
2. Писати `bot_run_events`.
3. Писати `bot_signals`.
4. Писати audit events.
5. Не відкривати exchange orders.

Acceptance criteria:

- run створює `bot_run_events` і `bot_signals`;
- invalid signal не публікується;
- signal-only safety гарантії покриті тестами.

### Етап 24. Bot Platform signal event publisher

Мета: додати downstream event publishing для вже persisted signals як окремий контрольований boundary.

1. Не змішувати цей компонент з `PersistentSignalPublisher`, який відповідає за persistence/audit у БД.
2. Додати `SignalEventPublisher` або outbox publisher для downstream signal events.
3. Публікувати тільки persisted valid signals.
4. Додати idempotency key.
5. Додати retry/backoff.
6. Додати metrics/logs.

Acceptance criteria:

- повторна публікація duplicate-safe;
- Redis failure не втрачає persisted signal;
- event publisher можна вимкнути env flag-ом;
- persisted signal залишається джерелом істини, навіть якщо downstream publish тимчасово недоступний.

### Етап 25. Control Panel instance UI

Мета: підключити lifecycle UI після готового bot-platform API.

1. Додати UI create/list bot instances.
2. Додати enable/pause controls.
3. Додати manual run action.
4. Показувати run status і validation errors.

Acceptance criteria:

- UI не читає bot tables напряму;
- всі write operations проходять через `control-panel/api` facade і `bot-platform-service`;
- ручний run видно у БД і UI status.

### Етап 26. Full Docker smoke

Мета: перевірити весь production-like flow.

1. Підняти infra, control-panel, market-data, bot-platform API, bot-platform runner.
2. Застосувати migrations.
3. Запустити `bot_modules_sync`.
4. Дочекатися healthy services.
5. Перевірити module list у control-panel/API.
6. Створити bot instance.
7. Запустити manual signal-only run.
8. Перевірити `bot_runs`, `bot_signals`, Redis signal event.

Acceptance criteria:

- весь сценарій проходить однією documented командою або smoke script;
- legacy bot services не стартують;
- повторний запуск smoke не ламає стан і не створює неконтрольовані дублікати.

## 5. Ризики

- Long-running scheduler може дублювати ingest без lock/idempotency.
- Redis outbox publisher може створювати duplicate events без idempotent publish.
- Healthcheck може бути занадто поверхневим і показувати healthy для неробочого сервісу.
- Автоматичні migrations у service startup можуть ламати rolling deploy.
- Control-panel direct DB reads можуть закріпити internal schema як public contract.
- Bot runner без freshness checks може генерувати signals на старих snapshots.

## 6. Мінімальний Definition of Done для першого milestone

Перший production-ready milestone фокусується на `market-data-service` як постійному backend-сервісі та базовій сумісності з `bot-platform-service` jobs. Він вважається виконаним, якщо:

- `market_data_migrate` застосовує migrations;
- `market_data` стартує як постійний container;
- `/health/ready` повертає 200 після старту;
- scheduler пише хоча б один валідний market snapshot;
- outbox publisher публікує event у Redis Stream;
- `docker stop market_data` завершує процес без stack trace;
- `bot_platform_migrate` і `bot_modules_sync` проходять після цього без ручних дій;
- документація описує локальний запуск, Docker запуск, перевірку health і типові помилки.
