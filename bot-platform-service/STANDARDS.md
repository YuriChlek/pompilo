# Bot Platform Service Standards

## Purpose

This document defines the required naming, structural, and architectural conventions for the Python `bot-platform-service`.

The goal is consistency:

- bot module ownership is explicit;
- platform orchestration stays separated from strategy logic;
- bot modules consume immutable market snapshots instead of legacy candle tables;
- bot modules publish standardized trading signals, not order intents or exchange orders;
- configuration, state, permissions, and secrets are isolated per bot instance;
- production behavior remains testable, observable, idempotent, and rollback-friendly.

## Stack Rules

The service uses:

- Python 3.12+
- PostgreSQL
- SQLAlchemy Core 2.0
- SQLAlchemy async engine with `asyncpg`
- Alembic for migrations

Rules:

- Use SQLAlchemy Core as the default query builder and schema expression layer.
- Do not use classic SQLAlchemy ORM in platform persistence.
- Use Alembic as the only supported way to change database schema.
- Keep raw SQL inside repository/query modules only, and only when SQLAlchemy Core is not precise or efficient enough.
- Business logic must not import `asyncpg`, SQLAlchemy engines, SQLAlchemy tables, or raw SQL directly.
- All monetary, price, quantity, confidence, and volume values use `Decimal` in Python and `NUMERIC` in PostgreSQL.
- Never convert trading, price, quantity, or signal confidence values to `float` in persistence or domain logic.
- Bot Platform is signal-only. It must not open positions, create exchange orders, manage fills, or call private exchange execution APIs.
- Bot modules may use well-maintained third-party analytical libraries such as
  `pandas`, `pandas-ta`, `ta`, or similar packages for indicators and time-series
  calculations when that is more reliable than reimplementing the algorithm. Add such
  dependencies explicitly to project dependency metadata, keep them out of
  lightweight-import files, and normalize their outputs back into platform domain models.

## Naming Rules

### General

- This file is named `STANDARDS.md`; use the same spelling for Bot Platform documentation.
- Use `snake_case` for folders and file names.
- Use `PascalCase` for classes, dataclasses, protocols, exceptions, and enums.
- Use `snake_case` for functions, methods, variables, module names, and table names.
- Use `SCREAMING_SNAKE_CASE` only for true module-level constants.
- Use explicit names over abbreviations: `bot_instance`, not `inst`; `timeframe`, not `tf`; `signal_key`, not `sig_key`.

### Required file suffixes

- Domain model files: `models.py` or `*_models.py`
- Domain enum files: `enums.py` or `*_enums.py`
- Application service files: `*_service.py`
- Application port files: `ports.py` or `*_ports.py`
- Repository files: `*_repository.py`
- SQLAlchemy table definitions: `*_tables.py`
- Query helpers: `*_queries.py`
- Internal infrastructure adapters: `*_adapter.py`
- Event contracts: `events.py` or `*_events.py`
- Configuration files: `*_config.py`
- Pure utilities: descriptive `snake_case.py`, for example `timeframe_aliases.py`
- Tests: `test_*.py`

Do not use extra dots inside Python module names. Use `module_registry.py`, not `module.registry.py`.

### Bot module IDs

New platform-native bot module IDs must be short `snake_case` identifiers and must not end
with `_bot`.

Examples:

- use `spot_grid`, not `spot_grid_bot`;
- use `spot_greenwich`, not `spot_greenwich_bot`.

Legacy IDs with `_bot` may appear only in migration mapping, rollout comparison, audit
history, or rollback documentation. New production instances must use suffix-free module
IDs.

## Project Structure

New code should follow this structure:

```text
bot-platform-service/
├── STANDARDS.md
├── pyproject.toml
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
├── src/
│   └── bot_platform_service/
│       ├── __init__.py
│       ├── main.py
│       ├── domain/
│       ├── application/
│       ├── infrastructure/
│       │   ├── market_data/
│       │   └── signals/
│       ├── trading_bots/
│       │   ├── __init__.py
│       │   ├── spot_grid/
│       │   │   ├── __init__.py
│       │   │   ├── manifest.py
│       │   │   ├── adapter.py
│       │   │   ├── bot_config.py
│       │   │   ├── config_schema.py
│       │   │   ├── domain/
│       │   │   ├── application/
│       │   │   └── infrastructure/
│       │   └── spot_greenwich/
│       │       ├── __init__.py
│       │       ├── manifest.py
│       │       ├── adapter.py
│       │       ├── bot_config.py
│       │       ├── config_schema.py
│       │       ├── domain/
│       │       ├── application/
│       │       └── infrastructure/
│       ├── persistence/
│       ├── registry/
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

### `trading_bots/` package template

`src/bot_platform_service/trading_bots/` is the canonical ownership boundary for
platform-native trading bot modules.

Each module package must use this top-level template:

```text
trading_bots/<module_id>/
├── __init__.py
├── manifest.py
├── adapter.py
├── bot_config.py
├── config_schema.py
├── domain/
├── application/
└── infrastructure/
```

Rules:

- `manifest.py` owns module metadata and must be lightweight-import safe.
- `bot_config.py` owns module-local default strategy configuration as Python
  constants, dataclasses, parsers, and validators. Every bot with strategy
  parameters must have this file.
- `config_schema.py` owns machine-readable configuration schema metadata and must be
  lightweight-import safe.
- `config_schema.py` must expose JSON-safe metadata with a positive `schema_version`,
  non-empty `sections`, and section `fields`.
- Supported config field types are `string`, `integer`, `decimal`, `boolean`, `enum`,
  `symbol`, `symbol_list`, `timeframe`, `timeframe_list`, `secret_ref`, `object`, and
  `array`.
- Decimal field bounds and defaults should use strings in schema metadata, not Python
  floats.
- `config_schema.py` must mirror the defaults and bounds from `bot_config.py`; it may
  import `bot_config.py` only when that module remains lightweight and side-effect-free.
- `adapter.py` is the package-level Bot Platform entrypoint and implements the
  `BotModule` contract.
- `adapter.py` is an intentional exception to the broader `*_adapter.py` naming
  convention because the package name scopes the adapter.
- Internal infrastructure adapters inside `trading_bots/<module_id>/infrastructure/`
  must keep descriptive `*_adapter.py` file names.
- Module `domain/` code must not import platform infrastructure, SQLAlchemy, exchange
  clients, or root-level legacy bot packages.
- Module `application/` code orchestrates one platform run through ports and domain
  rules; it must not own schedulers, direct database queries, market sync, or exchange
  execution.
- Module `infrastructure/` code adapts platform capabilities to module-local ports and
  must remain signal-only.

### Bot-owned configuration

Bot-owned strategy configuration belongs in `trading_bots/<module_id>/bot_config.py`.

Rules:

- `bot_config.py` must be a pure Python module with no environment reads, file reads,
  database access, network access, exchange clients, Redis clients, or legacy root
  package imports.
- Do not use `os.getenv`, `.env` files, process environment variables, or Docker env
  variables for bot strategy defaults or thresholds.
- Use typed dataclasses and `Decimal` defaults for strategy prices, quantities,
  notionals, percentages, confidence values, and volumes.
- Persisted per-instance overrides come from `BotInstanceConfig.config` and are
  validated against `config_schema.py`; the adapter or application layer merges those
  overrides with `bot_config.py` defaults before invoking domain planning.
- Secrets must never be defaulted in `bot_config.py`. Use `secret_ref` fields in
  `config_schema.py` and resolve them through `SecretProvider` only outside domain code.
- Domain code receives a resolved config object explicitly. Domain code must not import
  platform persistence or read runtime configuration by itself.

### Strategy libraries

Do not hand-roll indicators, volatility metrics, or time-series transforms when a
well-maintained library can provide the same calculation with clearer behavior and tests.

Rules:

- Allowed examples include `pandas`, `pandas-ta`, `ta`, `stock-indicators`,
  NumPy-based helpers, or other focused analytical packages approved for the service
  dependency set.
- Add every new library to `pyproject.toml` or the relevant lock/dependency metadata; do
  not rely on undeclared transitive dependencies.
- Do not import heavy analytical libraries from `manifest.py` or `config_schema.py`.
  Those files must stay lightweight-import safe.
- Keep provider, database, Redis, network, and exchange SDK dependencies out of pure
  strategy code.
- Convert library outputs at the boundary. Domain-facing models must still use `Decimal`
  for prices, quantities, notionals, percentages, confidence values, and volumes.
- Add fixture tests for indicator values so future library upgrades cannot silently
  change strategy behavior.

## Layer Responsibilities

### `domain/`

Owns pure platform concepts and rules.

Allowed:

- bot module contracts;
- bot manifest models;
- bot instance config models;
- runtime context models;
- market snapshot DTOs;
- signal DTOs;
- run result DTOs;
- deterministic signal key and payload hash rules;
- timeframe alias normalization;
- enums for modes, statuses, permissions, signal types, and health states.

Forbidden:

- database access;
- provider API calls;
- Redis or queue access;
- SQLAlchemy imports;
- asyncpg imports;
- imports from concrete bot modules such as `spot_grid_bot` or `spot-greenwich-bot`;
- exchange clients or private exchange API calls;
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

- Application services orchestrate domain rules, repositories, registry services, and runtime ports.
- Application services depend on port protocols or repository interfaces, not concrete bot adapters or provider SDKs.
- Transaction boundaries are explicit in write use cases.
- Application services may decide when to validate, run, pause, resume, or disable a bot instance.
- Application services may decide when to publish a signal, but `SignalPublisher` owns persistence details.
- Application services must not import concrete legacy bots directly.

### `registry/`

Owns bot module discovery and manifest validation.

Rules:

- Registry stores metadata for available bot modules.
- Registry validates manifest shape, supported modes, required timeframes, and adapter path.
- Registry must not execute bot strategy code during metadata registration.
- Duplicate module registration must be idempotent.
- Platform-native adapter import paths must point to package-local `adapter.py` modules
  under `bot_platform_service.trading_bots.<module_id>.adapter`.
- Registry discovery for platform-native modules must import only `manifest.py` and
  `config_schema.py`; it must not import `adapter.py` during metadata validation.
- Registry validation must reject new module IDs ending in `_bot`.

### `trading_bots/`

Owns platform-native trading modules.

Typical contents:

```text
trading_bots/
├── spot_grid/
│   ├── manifest.py
│   ├── adapter.py
│   ├── config_schema.py
│   ├── domain/
│   ├── application/
│   └── infrastructure/
└── spot_greenwich/
    ├── manifest.py
    ├── adapter.py
    ├── config_schema.py
    ├── domain/
    ├── application/
    └── infrastructure/
```

Rules:

- New platform-native trading bot modules live here, not under service-level
  infrastructure packages.
- Module packages must use suffix-free IDs such as `spot_grid` and `spot_greenwich`.
- Platform-native modules must consume `BotMarketSnapshot` or `BotMarketDataContext`
  only through platform runtime capabilities.
- Platform-native modules must return standardized signals, diagnostics, notifications,
  and state changes through `BotRunResult`.
- Platform-native modules must not import root-level legacy bot packages after migration.
- Platform-native modules must not create exchange orders, private exchange clients,
  candle sync jobs, or standalone schedulers.

### `infrastructure/`

Owns external systems and concrete adapters.

Typical contents:

```text
infrastructure/
├── market_data/
├── notifications/
├── queues/
├── clocks/
└── health/
```

Rules:

- Market data adapters read `market-data-service` snapshots and normalize them into platform DTOs.
- Infrastructure adapters do not contain platform lifecycle policy.
- Service-level concrete bot adapters do not live under `infrastructure/`; bot runtime
  adapters live only in `trading_bots/<module_id>/adapter.py`.
- Infrastructure adapters must not create exchange orders or construct private exchange
  clients in platform mode.

### Bot-owned persistence

Platform-native bot modules should use platform `StateStore` for simple per-instance
runtime state.

When a module needs queryable module-owned persistence that does not fit `StateStore`,
rules are:

- use SQLAlchemy Core table definitions and repositories;
- do not use classic SQLAlchemy ORM unless service-wide standards change first;
- keep SQLAlchemy Core table definitions and repositories in the module
  `infrastructure/` package or in shared Bot Platform persistence modules, not in
  module `domain/`;
- use Alembic migrations for schema changes;
- keep monetary, price, quantity, volume, and confidence values as `Decimal` in Python
  and `NUMERIC` in PostgreSQL;
- do not reuse legacy bot DB helpers, standalone table creation scripts, or direct
  `psycopg2` access;
- do not let strategy or domain code import SQLAlchemy engines, tables, rows, or
  repositories directly.

### `persistence/`

Owns PostgreSQL access.

Typical contents:

```text
persistence/
├── db.py
├── tables/
│   ├── bot_modules_tables.py
│   ├── bot_instances_tables.py
│   ├── bot_runs_tables.py
│   ├── bot_signals_tables.py
│   └── bot_audit_events_tables.py
├── repositories/
│   ├── bot_module_repository.py
│   ├── bot_instance_repository.py
│   ├── bot_run_repository.py
│   ├── bot_signal_repository.py
│   └── bot_audit_event_repository.py
└── queries/
```

Rules:

- SQLAlchemy Core table definitions live only under `persistence/tables/`.
- All direct database access lives in `persistence/repositories/`.
- Repositories may import SQLAlchemy Core tables and compose queries.
- Repositories must not import concrete bot modules or execute bot strategy code.
- Repositories must not call exchange APIs, provider sync APIs, or notification SDKs.
- Raw SQL, if needed, lives in `persistence/queries/` or private repository helpers with tests.
- Repositories return domain/application DTOs, not SQLAlchemy rows leaking through the application layer.

### `workers/`

Owns long-running processes and queue consumers.

Typical workers:

- polling scheduler worker;
- market snapshot event consumer;
- bot run worker;
- health check worker;
- signal outbox publisher worker, if an outbox is introduced.

Rules:

- Workers are thin entrypoints.
- Workers call application services.
- Workers do not implement domain rules or SQL directly.
- Workers must be restart-safe and idempotent.
- A failed bot instance must not crash unrelated bot instances or the whole platform.

### `config/`

Owns environment parsing and service configuration.

Rules:

- Validate required environment variables at startup.
- Convert env strings into typed configuration once.
- Do not read `os.getenv` outside config modules.
- Keep database, queue, scheduler, registry, permissions, and observability settings separate.
- Do not store bot secrets or exchange credentials in service config.

### `observability/`

Owns logging, metrics, tracing, and health payload helpers.

Rules:

- Logs must include `module_id`, `instance_id`, `run_id`, `mode`, `symbol`, `timeframe`, `snapshot_id`, `status`, and `correlation_id` where available.
- Logs must not include API keys, Telegram tokens, exchange credentials, or raw secret values.
- Metrics names must be stable and documented before dashboard use.

## Database Conventions

### Schema

All Bot Platform tables live under:

```text
_bot_platform
```

Expected tables:

```text
bot_modules
bot_instances
bot_instance_configs
bot_runtime_state
bot_runs
bot_run_events
bot_signals
bot_permissions
bot_secrets_refs
bot_health_checks
bot_audit_events
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
- Use unique constraints for idempotency keys such as `signal_key`.
- Use optimistic locking for mutable runtime state.
- Avoid hidden implicit transactions. Write use cases must pass or open explicit transaction boundaries.

## Bot Module Contract Rules

### Manifest

Every bot module must provide a manifest with:

- `module_id`;
- `display_name`;
- `version`;
- `supported_modes`;
- `required_timeframes`;
- `required_market_data`;
- `supports_multi_symbol`;
- `config_schema_version`.

Rules:

- Secrets must never appear in manifest data.
- Manifest validation must happen before module registration.
- Manifest registration must not run strategy logic.

### Runtime context

Platform passes only approved capabilities to bot modules:

- `MarketDataSnapshotProvider`;
- `SignalPublisher`;
- `StateStore`;
- `NotificationPublisher`;
- `SecretProvider`;
- `StructuredLogger`;
- `MetricsRecorder`;
- `Clock`.

Rules:

- Capabilities are permission-scoped per instance.
- Missing permission must produce a typed permission error.
- `SignalPublisher` and `NotificationPublisher` use separate permissions.
- `SecretProvider` resolves only secret refs for the current `instance_id`.
- Bot modules must not create uncontrolled DB, market-data sync, exchange, or notification clients in platform mode.

### Signals

Rules:

- `BotSignal` is a trading signal, not an order and not an order intent.
- `BotSignal` must include deterministic `signal_key`.
- `payload_json` must be versioned through `payload_schema` and `payload_schema_version`.
- `payload_hash` must be calculated from canonical JSON with sorted keys and Decimal/string normalization.
- Duplicate publish with the same `signal_key` must be idempotent and return the existing `signal_id`.
- Future Execution/Trading Service may consume signals, but only that separate service may create order intents or exchange orders.

## Market Data Rules

Rules:

- Bot Platform must read market data through immutable Market Data Service snapshots.
- Bot Platform must not download candles from Binance or any other provider.
- Bot Platform must not read legacy bot candle tables in platform mode.
- Snapshot provider reads `market_snapshots` and `market_snapshot_candles`, then reconstructs candles in `ordinal` order.
- Snapshot provider must reject incomplete, stale, missing, or unsupported timeframe data with typed errors.
- Timeframe aliases are normalized centrally: `h1`, `1H`, `H1` to `1h`; `h4`, `4H`, `H4` to `4h`; `d1`, `1D`, `D1` to `1d`.

## Event and Queue Conventions

### Events

Rules:

- Events are versioned.
- Events include `event_id`, `occurred_at`, and `idempotency_key`.
- Events include enough identifiers to replay bot run inputs.
- Events do not include raw secret values.
- `CandleBatchReady` duplicates must not create duplicate bot runs or duplicate signals for the same instance and snapshot.

### Signal publishing

Rules:

- Signal persistence must be idempotent.
- A persisted signal must have a run event and audit trail.
- `PUBLISHED` means the platform accepted the signal, not that an execution service acted on it.

## Testing Standards

### Unit tests

Use unit tests for:

- manifest validation;
- timeframe alias normalization;
- deterministic signal key construction;
- payload hash construction;
- permission checks;
- run result DTO construction;
- bot lifecycle state transitions.

### Integration tests

Use integration tests for:

- Alembic migrations;
- SQLAlchemy Core repositories;
- idempotent module registration;
- idempotent signal publishing;
- runtime state optimistic locking;
- market snapshot replay from `market_snapshots` and `market_snapshot_candles`.

### Contract tests

Use contract tests for:

- `BotModule` protocol conformance;
- manifest conformance;
- market-data DTO conformance;
- signal DTO conformance;
- `BotRunResult` structure;
- no direct DB, exchange, or market-data client construction in platform mode;
- permission scoping for `SignalPublisher`, `NotificationPublisher`, `StateStore`, and `SecretProvider`.

### Worker smoke tests

Use smoke tests for:

- duplicate `CandleBatchReady` delivery;
- missing snapshot;
- incomplete snapshot;
- stale snapshot;
- bot run failure isolation;
- platform restart recovery.

## Prohibited Patterns

- Importing concrete bot modules in `domain/`, `application/`, `registry/`, or `persistence/`.
- Creating new platform-native bot module IDs ending in `_bot`.
- Direct Binance candle sync from a bot module in platform mode.
- Fallback from platform snapshot provider to legacy bot candle tables.
- Storing secrets in manifest or config JSON.
- Opening positions or creating exchange orders in Bot Platform.
- Live execution in Bot Platform.
- Creating private exchange clients in platform-mode adapters or runtime context.
- Treating `BotSignal` as execution approval.
- Treating `BotSignal` as an order intent.
- Unversioned free-form `BotSignal.payload`.
- Non-deterministic signal IDs or publish without an idempotency key.
- Free-form `BotRunResult` without structured signals, notifications, state changes, and redacted errors.
- Shared mutable global config between instances.
- Bot module reading another instance's config, state, or secrets.
- Adding new bot adapters without contract tests.

## Review Checklist

Before merging Bot Platform changes, verify:

- folder placement matches this document;
- SQL lives only in persistence modules;
- migrations are Alembic-based;
- platform-native bot modules live under `trading_bots/`;
- no `bot_platform_service` runtime code imports root-level legacy bot packages;
- market data is read from snapshots, not legacy candle tables;
- signals have deterministic keys and versioned payloads;
- signal publishing is idempotent;
- runtime context capabilities are permission-scoped;
- no platform-mode path can create private exchange clients or exchange orders;
- no secrets can appear in logs, manifests, configs, or audit payloads;
- failed bot instances are isolated from other instances.
