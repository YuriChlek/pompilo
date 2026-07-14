# Market Data Service Standards

## Purpose

This document defines the required naming, structural, and architectural conventions for the Python `market-data-service`.

The goal is consistency:

- market data ownership is explicit;
- data ingestion, validation, persistence, and event publishing stay separated;
- PostgreSQL access is centralized and auditable;
- future consumers can consume immutable snapshots instead of ad-hoc candle ranges;
- production behavior remains testable, observable, and rollback-friendly.

## Stack Rules

The service uses:

- Python 3.12+
- PostgreSQL
- SQLAlchemy Core 2.0
- SQLAlchemy async engine with `asyncpg`
- Alembic for migrations

Rules:

- Use SQLAlchemy Core as the default query builder and schema expression layer.
- Do not use classic SQLAlchemy ORM in the ingestion hot path.
- Use Alembic as the only supported way to change database schema.
- Keep raw SQL inside repository/query modules only, and only when SQLAlchemy Core is not precise or efficient enough.
- Business logic must not import `asyncpg`, SQLAlchemy engines, SQLAlchemy tables, or raw SQL directly.
- All monetary, price, quantity, and volume values use `Decimal` in Python and `NUMERIC` in PostgreSQL.
- Never convert price or volume data to `float` in persistence or domain logic.

## Naming Rules

### General

- This file is named `STANDARDS.md`; use the same spelling for new Market Data Service documentation.
- Use `snake_case` for folders and file names.
- Use `PascalCase` for classes, dataclasses, protocols, exceptions, and enums.
- Use `snake_case` for functions, methods, variables, module names, and table names.
- Use `SCREAMING_SNAKE_CASE` only for true module-level constants.
- Use explicit names over abbreviations: `market_snapshot`, not `mkt_snap`; `timeframe`, not `tf`.

### Required file suffixes

- Domain model files: `models.py` or `*_models.py`
- Domain enum files: `enums.py` or `*_enums.py`
- Application service files: `*_service.py`
- Application port files: `ports.py` or `*_ports.py`
- Repository files: `*_repository.py`
- SQLAlchemy table definitions: `*_tables.py`
- Query helpers: `*_queries.py`
- Provider adapters: `*_adapter.py`
- Event contracts: `events.py` or `*_events.py`
- Configuration files: `*_config.py`
- Pure utilities: descriptive `snake_case.py`, for example `timeframe_window.py`
- Tests: `test_*.py`

Do not use extra dots inside Python module names. Use `candle_repository.py`, not `candle.repository.py`.

## Project Structure

New code should follow this structure:

```text
market-data-service/
├── STANDARDS.md
├── pyproject.toml
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
├── src/
│   └── market_data_service/
│       ├── __init__.py
│       ├── main.py
│       ├── domain/
│       ├── application/
│       ├── infrastructure/
│       ├── persistence/
│       ├── workers/
│       ├── config/
│       └── observability/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   └── fixtures/
└── README.md
```

Not every folder needs to exist on day one, but new code must follow this ownership model when the concern exists.

## Layer Responsibilities

### `domain/`

Owns pure market-data concepts and rules.

Allowed:

- canonical symbol value objects;
- timeframe rules;
- closed-candle validation;
- freshness and staleness rules;
- gap detection;
- snapshot completeness rules;
- event contract dataclasses;
- enums for statuses and source types.

Forbidden:

- database access;
- provider API calls;
- Redis or queue access;
- SQLAlchemy imports;
- asyncpg imports;
- logging side effects inside pure rules unless explicitly passed as a dependency.

### `application/`

Owns orchestration use cases and ports.

Typical contents:

```text
application/
├── services/
├── ports/
├── commands/
└── dto/
```

Rules:

- Application services orchestrate domain rules and repositories.
- Application services depend on port protocols or repository interfaces, not concrete provider SDKs.
- Transaction boundaries are explicit in write use cases.
- Application services may decide what to do, but not how SQL is written.
- Application services may decide when to publish an outbox event, but the repository writes it.

### `infrastructure/`

Owns external systems.

Typical contents:

```text
infrastructure/
├── providers/
│   └── binance_spot_adapter.py
├── queues/
├── clocks/
└── health/
```

Rules:

- Provider adapters normalize external API responses into domain/application DTOs.
- Provider adapters do not write to the database.
- Provider adapters do not know about consumers or readiness workflows.
- Private exchange execution credentials do not belong here.
- Market-data provider API keys, if ever needed, must be separate from trading execution keys.

### `persistence/`

Owns PostgreSQL access.

Typical contents:

```text
persistence/
├── db.py
├── tables/
│   ├── market_symbols_tables.py
│   ├── market_candles_tables.py
│   ├── market_snapshots_tables.py
│   └── outbox_events_tables.py
├── repositories/
│   ├── market_symbol_repository.py
│   ├── provider_symbol_repository.py
│   ├── candle_repository.py
│   ├── batch_repository.py
│   ├── snapshot_repository.py
│   └── outbox_repository.py
└── queries/
```

Rules:

- SQLAlchemy Core table definitions live only under `persistence/tables/`.
- All direct database access lives in `persistence/repositories/`.
- Repositories may import SQLAlchemy Core tables and compose queries.
- Repositories must not call provider APIs, publish queue messages, or contain trading strategy logic.
- Raw SQL, if needed, lives in `persistence/queries/` or private repository helpers with tests.
- Repositories return domain/application DTOs, not SQLAlchemy rows leaking through the application layer.

### `workers/`

Owns long-running processes and queue consumers.

Typical workers:

- sync scheduler worker;
- market data sync worker;
- backfill worker;
- outbox publisher worker;

Rules:

- Workers are thin entrypoints.
- Workers call application services.
- Workers do not implement domain rules or SQL directly.
- Workers must be restart-safe and idempotent.

### `config/`

Owns environment parsing and service configuration.

Rules:

- Validate required environment variables at startup.
- Convert env strings into typed configuration once.
- Do not read `os.getenv` outside config modules.
- Keep provider, database, queue, scheduler, and freshness settings separate.

### `observability/`

Owns logging, metrics, tracing, and health payload helpers.

Rules:

- Logs must include `source`, `canonical_symbol`, `timeframe`, `batch_id`, `snapshot_id`, and `correlation_id` where available.
- Logs must not include provider secrets.
- Metrics names must be stable and documented before dashboard use.

## Database Conventions

### Schema

All market data tables live under:

```text
_market_data
```

Expected tables:

```text
market_symbols
provider_symbols
market_candles
market_data_batches
market_snapshots
market_snapshot_candles
sync_jobs
outbox_events
provider_health
```

### Migrations

Rules:

- Use Alembic for every schema change.
- Do not manually edit a production database outside Alembic.
- Do not make unrelated schema changes in the same migration.
- Migration names must be descriptive and timestamped by Alembic.
- Migrations must be reversible unless data retention or safety requirements explicitly forbid it.
- For irreversible migrations, include an explicit comment explaining the rollback policy.

### SQLAlchemy Core

Rules:

- Table definitions must be explicit and reviewed like API contracts.
- Use PostgreSQL dialect inserts for `ON CONFLICT` handling.
- Keep bulk duplicate-safe candle insert in one repository method with tests.
- Keep advisory lock logic in repository code and document the lock key.
- Avoid hidden implicit transactions. Write use cases must pass or open explicit transaction boundaries.

## Event and Queue Conventions

### Event contracts

Event contracts live in:

```text
domain/events/
```

Required events:

- `CandleBatchReady`

Rules:

- Events are versioned.
- Events include `event_id`, `occurred_at`, and `idempotency_key`.
- Events include enough snapshot identifiers to replay market data input.
- Events do not include raw candle arrays.

### Outbox

Rules:

- Events are written to `outbox_events` in the same transaction as the state change they describe.
- Do not publish broker messages directly from a sync transaction.
- Outbox publisher may deliver duplicates.
- Consumers must be idempotent.
- `PUBLISHED` means the broker accepted the event, not that every consumer processed it.

## Testing Standards

### Unit tests

Use unit tests for:

- timeframe close boundary rules;
- closed-candle validation;
- gap detection;
- freshness policy;
- snapshot hash calculation;
- event idempotency key construction.

### Integration tests

Use integration tests for:

- Alembic migrations;
- SQLAlchemy Core repositories;
- bulk upsert behavior;
- advisory locks;
- transaction rollback;
- outbox idempotency;
- snapshot membership replay.

### Worker smoke tests

Use smoke tests for:

- duplicate sync jobs;
- provider timeout;
- partial provider response;
- outbox publisher restart;
- stale snapshot blocking.

## Prohibited Patterns

- Business logic importing SQLAlchemy tables.
- Application services constructing raw SQL strings.
- Provider adapters writing to PostgreSQL.
- Workers implementing domain validation inline.
- Publishing readiness events before database commit.
- Reading candles by "latest range" when a `snapshot_id` is available.
- Converting Decimal/NUMERIC market data to float.
- Hardcoding provider symbol mappings outside `provider_symbols`.
- Mixing execution credentials with market-data provider configuration.

## Review Checklist

Before merging market-data-service changes, verify:

- folder placement matches this document;
- SQL lives only in persistence modules;
- migrations are Alembic-based;
- tests cover repository behavior for critical queries;
- new events have idempotency keys;
- new sync paths are restart-safe;
- no ready snapshot event can be created for incomplete, stale, or gap data;
- no provider credentials or secrets can appear in logs.
