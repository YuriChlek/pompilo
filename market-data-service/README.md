# Market Data Service

`market-data-service` - Python 3.12 сервіс централізованого збору, нормалізації, збереження та публікації market data для Pampilo trading platform.

Сервіс відповідає за symbol registry, candle ingestion, batch tracking, snapshots, gap detection, transactional outbox і scheduler contracts. Trading bots мають споживати market snapshots, а не читати legacy candle tables напряму.

## Поточний статус

Сервіс має production CLI entrypoint, runtime composition root, HTTP health/readiness/metrics API, scheduler lifecycle, outbox publisher lifecycle, one-shot maintenance jobs, Binance Spot provider, fixture provider для deterministic smoke, Redis Stream broker, SQLAlchemy Core persistence, Alembic migrations і тестове покриття.

Основний production режим - long-running process:

```bash
python -m market_data_service.main collect
```

Міграції не запускаються автоматично всередині long-running контейнера. Для них використовується окремий Alembic job `market_data_migrate`.

## Вимоги

- Python 3.12+
- PostgreSQL 16+
- Redis 7+
- Docker і Docker Compose v2, якщо запускаєш через контейнерну інфраструктуру
- Доступ до Binance public REST API для реального ingest

## Структура

```text
market-data-service/
├── alembic/                         # Alembic migrations
├── docs/                            # rollout, observability, load docs
├── src/market_data_service/
│   ├── application/                 # use cases, ports, services
│   ├── config/                      # DB, provider, queue, scheduler config
│   ├── domain/                      # pure market data models/rules/events
│   ├── infrastructure/              # provider and queue adapters
│   ├── observability/               # metrics, alerts, logs
│   ├── persistence/                 # SQLAlchemy Core tables/repositories
│   ├── runtime/                     # composition root, HTTP server, lifecycle
│   └── workers/                     # scheduler/outbox worker classes
├── scripts/                         # Docker smoke scripts
├── tests/
├── Dockerfile
├── pyproject.toml
└── alembic.ini
```

## Локальний запуск через venv

Виконуй команди з директорії сервісу:

```bash
cd market-data-service
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Якщо `python3.12` недоступний, перевір версію:

```bash
python3 --version
```

Сервіс очікує Python `>=3.12`.

## Змінні середовища

Мінімальна конфігурація PostgreSQL:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=admin
export DB_PASSWORD=admin_pass
export DB_NAME=pampilo_db
export DATABASE=pampilo_db
```

Альтернативно можна передати повний URL:

```bash
export MARKET_DATA_DATABASE_URL=postgresql+asyncpg://admin:admin_pass@localhost:5432/pampilo_db
```

`MARKET_DATA_DATABASE_URL` має пріоритет над `DATABASE_URL` і `DB_*`.

Redis config:

```bash
export MARKET_DATA_REDIS_URL=redis://localhost:6379/0
export MARKET_DATA_OUTBOX_STREAM=market-data-events
```

Provider config:

```bash
export MARKET_DATA_PROVIDER_MODE=binance
export BINANCE_REST_ENDPOINT=https://api.binance.com
export BINANCE_REQUEST_TIMEOUT_SECONDS=30
export BINANCE_KLINE_LIMIT=1000
export BINANCE_MAX_CONCURRENT_REQUESTS=8
```

Для локальних smoke-перевірок без зовнішнього Binance API можна використати deterministic fixture provider:

```bash
export MARKET_DATA_PROVIDER_MODE=fixture
export MARKET_DATA_PROVIDER_SYMBOLS=ETHUSDT
export MARKET_DATA_TIMEFRAMES=1h
export MARKET_DATA_SCHEDULER_JITTER_SECONDS=0
export MARKET_DATA_1H_SAFETY_DELAY_SECONDS=0
```

Collect config:

```bash
export MARKET_DATA_SOURCE=binance_spot
export MARKET_DATA_PROVIDER_SYMBOLS=ETHUSDT,SOLUSDT
export MARKET_DATA_TIMEFRAMES=1h,4h,1d
export MARKET_DATA_COLLECT_POLL_INTERVAL_SECONDS=30
export MARKET_DATA_COLLECT_ON_START=true
export MARKET_DATA_COLLECT_MAX_JOBS_PER_TICK=100
export MARKET_DATA_COLLECT_BOOTSTRAP_LOOKBACK_YEARS=2
export MARKET_DATA_COLLECT_BOOTSTRAP_MAX_CHUNKS_PER_TICK=20
export MARKET_DATA_PROVIDER_MAX_CONCURRENCY=3
export MARKET_DATA_PROVIDER_PRIORITY=binance,bybit
export MARKET_DATA_SCHEDULER_JITTER_SECONDS=30
```

Safety delay config:

```bash
export MARKET_DATA_1H_SAFETY_DELAY_SECONDS=30
export MARKET_DATA_4H_SAFETY_DELAY_SECONDS=45
export MARKET_DATA_1D_SAFETY_DELAY_SECONDS=90
```

HTTP/runtime config:

```bash
export MARKET_DATA_HTTP_HOST=0.0.0.0
export MARKET_DATA_HTTP_PORT=8010
export MARKET_DATA_SCHEDULER_ENABLED=true
export MARKET_DATA_OUTBOX_PUBLISHER_ENABLED=true
export MARKET_DATA_SHUTDOWN_TIMEOUT_SECONDS=30
```

## PostgreSQL і Redis локально

Найпростіше підняти infra з кореня репозиторію:

```bash
cd ..
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  up -d postgres redis
```

Перевірити статус:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  ps postgres redis
```

## Міграції бази даних

З директорії `market-data-service`:

```bash
source .venv/bin/activate
alembic upgrade head
```

Подивитися поточну ревізію:

```bash
alembic current
```

Подивитися історію:

```bash
alembic history
```

Створити нову migration після зміни SQLAlchemy Core tables:

```bash
alembic revision --autogenerate -m "describe_change"
```

Перед комітом завжди переглядай autogenerated migration вручну, особливо для partitions, indexes і constraints.

## Локальний запуск сервісу без Docker

Перед запуском переконайся, що PostgreSQL і Redis доступні, env vars експортовані, а Alembic migrations застосовані.

Запустити long-running candle collection service:

```bash
cd market-data-service
source .venv/bin/activate
python -m market_data_service.main collect
```

Після старту доступні endpoints:

```bash
curl http://127.0.0.1:8010/health/live
curl http://127.0.0.1:8010/health/ready
curl http://127.0.0.1:8010/metrics
curl 'http://127.0.0.1:8010/snapshots/latest?source=BINANCE_SPOT&symbol=ETHUSDT&timeframe=1h'
```

Docker healthcheck використовує той самий readiness contract через CLI:

```bash
python -m market_data_service.main healthcheck
```

Коректне завершення:

```bash
Ctrl+C
```

або, якщо процес запущений контейнером:

```bash
docker stop pampilo-platform-market_data-1
```

## CLI команди

Показати довідку:

```bash
python -m market_data_service.main --help
```

Перевірити конфігурацію БД:

```bash
python -m market_data_service.main db:check
```

Запустити long-running collect runtime:

```bash
python -m market_data_service.main collect
```

Синхронізувати registry symbols:

```bash
python -m market_data_service.main symbols:sync
```

Запустити один collection tick:

```bash
python -m market_data_service.main scheduler:run-once
```

Обробити наступний pending sync job, записати candles/batch/snapshot/outbox records:

```bash
python -m market_data_service.main sync:run-next
```

Опублікувати одну пачку pending outbox events у Redis Stream:

```bash
python -m market_data_service.main outbox:publish-once
```

Створити backfill requests:

```bash
python -m market_data_service.main backfill \
  --symbol ETHUSDT \
  --timeframe 1h \
  --from 2026-07-16T00:00:00Z \
  --to 2026-07-16T06:00:00Z
```

Просканувати gaps без запуску backfill:

```bash
python -m market_data_service.main gaps:scan \
  --symbol ETHUSDT \
  --timeframe 1h \
  --from 2026-07-16T00:00:00Z \
  --to 2026-07-16T06:00:00Z
```

Просканувати gaps і створити backfill requests:

```bash
python -m market_data_service.main gaps:scan \
  --symbol ETHUSDT \
  --timeframe 1h \
  --from 2026-07-16T00:00:00Z \
  --to 2026-07-16T06:00:00Z \
  --create-backfill
```

Replay outbox events:

```bash
python -m market_data_service.main outbox:replay --dry-run
python -m market_data_service.main outbox:replay --from-id 1 --to-id 100
```

Команди `scheduler`, `outbox-publisher` і `db:revision` зарезервовані для майбутнього окремого deployment mode і зараз повертають явний unsupported exit code.

## Локальний deterministic smoke без Docker runtime

Цей сценарій використовує fixture provider, тому не залежить від Binance:

```bash
export MARKET_DATA_PROVIDER_MODE=fixture
export MARKET_DATA_PROVIDER_SYMBOLS=ETHUSDT
export MARKET_DATA_TIMEFRAMES=1h
export MARKET_DATA_SCHEDULER_JITTER_SECONDS=0
export MARKET_DATA_1H_SAFETY_DELAY_SECONDS=0

alembic upgrade head
python -m market_data_service.main symbols:sync
python -m market_data_service.main collect --once
python -m market_data_service.main collect
```

## Тести

Усі тести:

```bash
pytest
```

Тільки unit:

```bash
pytest tests/unit
```

Тільки smoke:

```bash
pytest tests/smoke
```

Один файл:

```bash
pytest tests/unit/test_single_symbol_sync_service.py
```

## Docker image

З кореня репозиторію:

```bash
docker build -t platform_market_data_service ./market-data-service
```

Перевірити контейнер:

```bash
docker run --rm platform_market_data_service python -m market_data_service.main --help
```

## Docker Compose

У загальній Docker-інфраструктурі сервіс підключений через:

```text
infra/compose/docker-compose.platform.yaml
```

Запустити PostgreSQL, Redis і migration job:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  up --build postgres redis market_data_migrate
```

Запустити тільки `market_data` після migrations:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  up --build market_data
```

Запустити one-shot jobs:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  run --rm market_data_symbols_sync
```

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  run --rm market_data python -m market_data_service.main scheduler:run-once
```

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  run --rm market_data python -m market_data_service.main sync:run-next
```

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  run --rm market_data python -m market_data_service.main outbox:publish-once
```

Запустити platform profile повністю:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  up --build
```

Зупинити:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  down
```

Повний runtime smoke для market-data:

```bash
bash market-data-service/scripts/docker_runtime_smoke.sh
```

## Development workflow

Типовий цикл роботи:

```bash
cd market-data-service
source .venv/bin/activate
python -m pip install -e ".[test]"
pytest
alembic upgrade head
```

Якщо змінив persistence tables:

```bash
alembic revision --autogenerate -m "short_description"
alembic upgrade head
pytest tests/unit tests/smoke
```

Якщо змінив provider або scheduler logic:

```bash
pytest tests/unit/test_binance_spot_adapter.py
pytest tests/unit/test_market_data_scheduler_service.py
pytest tests/unit/test_market_data_scheduler_worker.py
pytest tests/smoke/test_stage_12_market_data_scheduler.py
```

Якщо змінив outbox або Redis integration:

```bash
pytest tests/unit/test_outbox_publisher_service.py
pytest tests/unit/test_redis_stream_broker.py
pytest tests/smoke/test_stage_10_transactional_outbox.py
pytest tests/smoke/test_stage_11_outbox_publisher.py
```

## Архітектурні правила

Перед змінами прочитай:

- `STANDARDS.md`
- `docs/production_rollout.md`
- `docs/observability_baseline.md`
- `docs/load_rate_limit_hardening.md`

Ключові правила:

- domain layer не імпортує infrastructure, SQLAlchemy, asyncpg, Redis або provider SDK;
- persistence змінюється тільки через Alembic;
- всі price/quantity/volume значення використовують `Decimal`;
- market data service не відкриває угоди і не виконує trading decisions;
- legacy trading bots не входять у Docker runtime цього сервісу.
