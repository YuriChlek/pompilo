# Bot Platform Service

`bot-platform-service` - Python 3.12 сервіс для signal-only запуску trading bot modules у складі Pampilo trading platform.

Сервіс не відкриває позиції, не створює exchange orders і не працює з private exchange execution API. Його зона відповідальності - registry, lifecycle, runtime context, health/audit/state і публікація стандартизованих trading signals для майбутнього execution layer.

## Поточний статус

Сервіс має production CLI entrypoint, runtime composition root, HTTP health/readiness/metrics API, admin API для bot modules/config schemas/instances/manual runs, runner process, signal persistence, optional Redis signal event publisher, Alembic migrations, Docker image і тестове покриття.

Основні production режими:

```bash
python -m bot_platform_service.main serve
python -m bot_platform_service.main runner
```

Міграції і синхронізація bot modules запускаються окремими one-shot jobs: `bot_platform_migrate` і `bot_modules_sync`.

## Вимоги

- Python 3.12+
- PostgreSQL 16+
- Redis 7+, якщо увімкнена публікація downstream signal events
- Доступний `market-data-service` HTTP API для runner/manual run
- Docker і Docker Compose v2, якщо запускаєш через контейнерну інфраструктуру

## Структура

```text
bot-platform-service/
├── alembic/                         # Alembic migrations
├── docs/                            # rollout і production docs
├── src/bot_platform_service/
│   ├── application/                 # orchestration services
│   ├── config/                      # runtime config helpers
│   ├── domain/                      # pure contracts, DTOs, enums
│   ├── infrastructure/              # adapters
│   ├── observability/               # health, metrics, alerts
│   ├── persistence/                 # SQLAlchemy Core tables/repositories
│   ├── registry/                    # module discovery/resolution
│   ├── runtime/                     # composition root, HTTP server, health
│   ├── workers/                     # runner loop
│   └── trading_bots/                # platform-native bot modules
├── tests/
├── Dockerfile
├── pyproject.toml
└── alembic.ini
```

## Локальний запуск через venv

Виконуй команди з директорії сервісу:

```bash
cd bot-platform-service
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

Мінімальна конфігурація для локального PostgreSQL:

```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=admin
export DB_PASSWORD=admin_pass
export DB_NAME=pampilo_db
```

Альтернативно можна передати повний URL:

```bash
export BOT_PLATFORM_DATABASE_URL=postgresql+asyncpg://admin:admin_pass@localhost:5432/pampilo_db
```

`BOT_PLATFORM_DATABASE_URL` має пріоритет над `DB_*`.

HTTP runtime:

```bash
export BOT_PLATFORM_HTTP_HOST=0.0.0.0
export BOT_PLATFORM_HTTP_PORT=8092
```

Market Data integration:

```bash
export BOT_PLATFORM_MARKET_DATA_BASE_URL=http://localhost:8010
export BOT_PLATFORM_MARKET_DATA_SOURCE=BINANCE_SPOT
```

Runner config:

```bash
export BOT_PLATFORM_RUNNER_POLL_INTERVAL_SECONDS=15
```

Downstream signal events через Redis Stream:

```bash
export BOT_PLATFORM_SIGNAL_EVENTS_ENABLED=false
export BOT_PLATFORM_SIGNAL_EVENTS_REDIS_URL=redis://localhost:6379/0
export BOT_PLATFORM_SIGNAL_EVENTS_STREAM=bot-platform-signals
export BOT_PLATFORM_SIGNAL_EVENTS_MAX_RETRIES=3
export BOT_PLATFORM_SIGNAL_EVENTS_RETRY_BACKOFF_SECONDS=0.25
```

Для повного smoke або локальної перевірки Redis publisher встанови:

```bash
export BOT_PLATFORM_SIGNAL_EVENTS_ENABLED=true
```

## PostgreSQL, Redis і Market Data локально

Найпростіше підняти PostgreSQL/Redis з кореня репозиторію:

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

Для manual run/runner потрібен готовий `market-data-service`. Його можна запустити локальним Python у сусідній директорії або через Docker Compose:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  up --build market_data
```

## Міграції бази даних

З директорії `bot-platform-service`:

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

Перед комітом завжди переглядай autogenerated migration вручну.

## Локальний запуск сервісу без Docker

Перед запуском переконайся, що PostgreSQL доступний, Alembic migrations застосовані, `bot:modules:sync` виконаний, а `market-data-service` доступний для manual run/runner.

Застосувати migrations:

```bash
cd bot-platform-service
source .venv/bin/activate
alembic upgrade head
```

Синхронізувати platform-native bot modules:

```bash
python -m bot_platform_service.main bot:modules:sync
```

Запустити HTTP API:

```bash
python -m bot_platform_service.main serve
```

Після старту доступні endpoints:

```bash
curl http://127.0.0.1:8092/health/live
curl http://127.0.0.1:8092/health/ready
curl http://127.0.0.1:8092/metrics
curl http://127.0.0.1:8092/admin/bot-modules
curl http://127.0.0.1:8092/admin/bot-modules/spot_grid/config-schema
curl http://127.0.0.1:8092/admin/bot-instances
```

Docker healthcheck використовує той самий readiness contract через CLI:

```bash
python -m bot_platform_service.main healthcheck
```

Запустити runner process окремо від HTTP API:

```bash
python -m bot_platform_service.main runner
```

Runner читає enabled bot instances, перевіряє доступність market snapshots через `BOT_PLATFORM_MARKET_DATA_BASE_URL` і не блокує admin API.

Коректне завершення long-running процесів:

```bash
Ctrl+C
```

## CLI команди

Показати довідку:

```bash
python -m bot_platform_service.main --help
```

Синхронізувати metadata bot modules у `_bot_platform.bot_modules`:

```bash
python -m bot_platform_service.main bot:modules:sync
```

Запустити HTTP API:

```bash
python -m bot_platform_service.main serve
```

Запустити runner:

```bash
python -m bot_platform_service.main runner
```

Перевірити readiness локального HTTP API:

```bash
python -m bot_platform_service.main healthcheck
```

Перевірити кастомний readiness URL:

```bash
python -m bot_platform_service.main healthcheck --url http://127.0.0.1:8092/health/ready --timeout 2
```

## Мінімальний локальний сценарій

1. Підняти PostgreSQL/Redis.
2. Запустити migrations.
3. Запустити `bot:modules:sync`.
4. Запустити `market-data-service`.
5. Запустити `bot-platform-service` HTTP API.
6. За потреби запустити runner окремим процесом.

Команди:

```bash
alembic upgrade head
python -m bot_platform_service.main bot:modules:sync
python -m bot_platform_service.main serve
```

В іншому терміналі:

```bash
python -m bot_platform_service.main runner
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
pytest tests/unit/test_stage_4_module_registry.py
```

## Docker image

З кореня репозиторію:

```bash
docker build -t platform_bot_platform_service ./bot-platform-service
```

Перевірити контейнер:

```bash
docker run --rm platform_bot_platform_service python -m bot_platform_service.main --help
```

## Docker Compose

У загальній Docker-інфраструктурі сервіс підключений через:

```text
infra/compose/docker-compose.platform.yaml
```

Запустити PostgreSQL і migration job:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  up --build postgres bot_platform_migrate
```

Запустити sync job для bot modules:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  run --rm bot_modules_sync
```

Запустити Bot Platform HTTP API:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  up --build bot_platform
```

Запустити runner:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  --profile infra \
  --profile platform \
  up --build bot_platform_runner
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

Повний production-like smoke для market-data, bot-platform і control-panel:

```bash
infra/scripts/full-docker-smoke.sh
```

## Development workflow

Типовий цикл роботи:

```bash
cd bot-platform-service
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

## Архітектурні правила

Перед змінами прочитай:

- `STANDARDS.md`
- `docs/how_to_add_new_bot_module.md`
- `docs/production_readiness.md`

Ключові правила:

- domain layer не імпортує infrastructure, SQLAlchemy, asyncpg або concrete bot packages;
- persistence змінюється тільки через Alembic;
- всі price/quantity/confidence/volume значення використовують `Decimal`;
- platform service залишається signal-only;
- legacy standalone bots не підключаються до Docker-інфраструктури як runtime services.
