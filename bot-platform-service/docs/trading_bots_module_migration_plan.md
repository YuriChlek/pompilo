# Trading Bots Module Migration Plan

## Purpose

This document defines the target technical design for moving trading bot modules into `bot-platform-service`.

The current root-level bots remain untouched during planning:

- `spot_grid_bot`
- `spot-greenwich-bot`

They should be treated as legacy standalone implementations and rollback references until their platform-native replacements are complete and verified.

The new platform-native bots must live under:

```text
bot-platform-service/src/bot_platform_service/trading_bots/
```

New module identifiers must not use the `_bot` suffix.

## Target Module IDs

Use short platform-native module IDs:

| Legacy project | New platform module ID | New package path |
|---|---|---|
| `spot_grid_bot` | `spot_grid` | `bot_platform_service.trading_bots.spot_grid` |
| `spot-greenwich-bot` | `spot_greenwich` | `bot_platform_service.trading_bots.spot_greenwich` |

Legacy IDs such as `spot_grid_bot` and `spot_greenwich_bot` may be supported temporarily only for migration mapping, audit history, or rollback comparison. New platform instances should use `spot_grid` and `spot_greenwich`.

## Target Directory Layout

The `trading_bots` package is the ownership boundary for platform-native trading modules.
Both initial modules must use the same top-level template. The detailed files inside
`domain/`, `application/`, and `infrastructure/` may differ only because the strategies
have different complexity.

```text
bot-platform-service/
└── src/
    └── bot_platform_service/
        ├── application/
        ├── domain/
        ├── infrastructure/
        ├── persistence/
        ├── registry/
        ├── trading_bots/
        │   ├── __init__.py
        │   ├── spot_grid/
        │   │   ├── __init__.py
        │   │   ├── manifest.py
        │   │   ├── adapter.py
        │   │   ├── config_schema.py
        │   │   ├── domain/
        │   │   ├── application/
        │   │   └── infrastructure/
        │   └── spot_greenwich/
        │       ├── __init__.py
        │       ├── manifest.py
        │       ├── adapter.py
        │       ├── config_schema.py
        │       ├── domain/
        │       ├── application/
        │       └── infrastructure/
        └── workers/
```

Tests for these modules should remain in the service-level test tree so they can run with the rest of Bot Platform:

```text
bot-platform-service/tests/
├── unit/
│   ├── trading_bots/
│   │   ├── test_spot_grid_adapter.py
│   │   ├── test_spot_grid_domain.py
│   │   ├── test_spot_greenwich_adapter.py
│   │   └── test_spot_greenwich_domain.py
├── integration/
└── smoke/
```

## Module Template

Each bot package must expose a stable template.

```text
trading_bots/<module_name>/
├── __init__.py
├── manifest.py
├── adapter.py
├── config_schema.py
├── domain/
├── application/
└── infrastructure/
```

Each module should start with the same package-level template. The module may then add
strategy-specific files under the three internal layers.

Recommended internal shape:

```text
trading_bots/<module_name>/
├── domain/
│   ├── __init__.py
│   ├── models.py
│   ├── signals.py
│   └── *_policy.py / *_planner.py / *_models.py as needed
├── application/
│   ├── __init__.py
│   ├── ports.py
│   └── trading_cycle_service.py
└── infrastructure/
    ├── __init__.py
    └── platform_*_adapter.py / snapshot_*_provider.py as needed
```

Do not force both bots to have identical file names below the layer boundary. Force the
same ownership model and import direction instead.

`adapter.py` is the package-level module entrypoint for Bot Platform discovery and runtime
resolution. This is an intentional exception to the broader `*_adapter.py` file suffix
rule because the package name already scopes the adapter. Internal infrastructure adapters
inside `infrastructure/` should still use descriptive `*_adapter.py` names.

## Fit Against Current Bots

The template fits both current bots, but the amount of code per layer will differ.

### `spot_grid`

Current `spot_grid_bot` already separates:

- `domain/`: grid construction, regime detection, risk, portfolio allocation, target
  order building, runtime state, order diffs, and strategy config.
- `application/`: ports, trading cycle orchestration, analysis batch service,
  execution service, initialization, scheduler, dry-run formatting, and health.
- `infrastructure/`: Binance sync, Bybit clients, DB/state store, notifications,
  execution gateway, market-data provider, and live price monitor.

Platform-native `spot_grid` should move only the platform-safe parts:

- keep rich domain files such as planner, risk, portfolio, regime, target-order, and
  runtime-state modules under `trading_bots/spot_grid/domain/`;
- keep one platform trading-cycle orchestration under
  `trading_bots/spot_grid/application/`;
- replace venue execution, DB, sync, and notification infrastructure with platform
  adapters under `trading_bots/spot_grid/infrastructure/`;
- convert target-order decisions into `BotSignal` records in `adapter.py`;
- leave standalone scheduler, live order sync, candle sync, Dockerfile, and CLI runtime
  in the root-level legacy bot until explicit retirement.

### `spot_greenwich`

Current `spot-greenwich-bot` is smaller and already separates:

- `domain/`: models, Greenwich signals, planner, and execution policy.
- `application/`: ports, trading cycle orchestration, command dispatch, scheduler,
  runtime commands, execution service, and initialization.
- `infrastructure/`: Bybit execution, market-data provider/synchronizer, state store,
  and notifications.

Platform-native `spot_greenwich` should move only the platform-safe parts:

- keep models, signal generation, planner, and execution policy under
  `trading_bots/spot_greenwich/domain/`;
- keep platform one-run orchestration under `trading_bots/spot_greenwich/application/`;
- replace Bybit, DB, sync, and notification implementations with platform adapters under
  `trading_bots/spot_greenwich/infrastructure/`;
- convert execution decisions into `BotSignal` records in `adapter.py`;
- leave standalone command dispatch, scheduler, market-data sync, migrations, CLI, and
  direct exchange execution in the root-level legacy bot until explicit retirement.

### Shared Conclusion

The proposed structure is suitable because both bots already have clear domain,
application, and infrastructure boundaries. The migration should preserve those boundaries
while removing root-level imports and replacing all side effects with Bot Platform ports.

### `manifest.py`

`manifest.py` is the source of truth for module metadata.

Example:

```python
RAW_MANIFEST = {
    "module_id": "spot_grid",
    "display_name": "Spot Grid",
    "version": "1.0.0",
    "supported_modes": ("dry_run", "notification_only", "signal_only"),
    "required_timeframes": ("1h", "4h"),
    "required_market_data": ("snapshots",),
    "supports_multi_symbol": True,
    "config_schema_version": 1,
}

ADAPTER_PATH = "bot_platform_service.trading_bots.spot_grid.adapter"
ADAPTER_CLASS = "SpotGridAdapter"
```

Rules:

- `module_id` must match the package name.
- `module_id` must not end with `_bot`.
- `ADAPTER_PATH` must point to the package-local `adapter.py`.
- `ADAPTER_CLASS` must name the `BotModule` implementation inside `adapter.py`.
- Manifest data must not include secrets.
- Manifest data must be importable without importing exchange clients, DB clients, or heavy strategy runtime dependencies.

### `adapter.py`

`adapter.py` implements the Bot Platform `BotModule` contract.

Required behavior:

- `validate_config(config)` validates only platform config and module config schema.
- `initialize(context)` stores platform capabilities.
- `dry_run(request)` calculates signals without exchange execution or module-owned
  side effects. Platform persistence is controlled by the orchestration entrypoint.
- `run_once(request)` supports `notification_only` and `signal_only`.
- `start(request)` should return `START_NOT_SUPPORTED` unless a platform worker explicitly owns long-running execution.
- `stop(instance_id)` should be idempotent.
- `health(instance_id)` should report adapter readiness and dependency state.

The adapter may call package-local application services. It must not import root-level legacy bot packages after migration is complete.

### `config_schema.py`

`config_schema.py` owns versioned runtime config parsing.

Rules:

- Keep `config_schema_version` explicit.
- Reject unknown or incompatible schema versions.
- Normalize symbol, timeframe, risk, and portfolio settings once.
- Keep secrets as references only.
- Return typed config objects, not raw dictionaries.

### `domain/`

`domain/` contains strategy rules and signal decisions.

Allowed:

- pure dataclasses;
- signal generation;
- risk and portfolio decision logic;
- deterministic helpers;
- Decimal-based price, quantity, volume, and confidence values.

Forbidden:

- database access;
- exchange clients;
- Telegram or notification SDKs;
- direct market-data provider calls;
- `os.getenv`;
- root-level legacy bot imports.

### `application/`

`application/` orchestrates one platform run using ports and domain rules.

Allowed:

- trading cycle services;
- module-local ports;
- request-to-domain mapping;
- domain-to-signal mapping coordination.

Forbidden:

- private exchange execution;
- direct database queries;
- direct market sync or backfill;
- global scheduler ownership.

### `infrastructure/`

`infrastructure/` adapts platform capabilities to module-local ports.

Allowed:

- adapters around `BotMarketDataContext`;
- adapters around `MarketDataSnapshotProvider`;
- adapters around `StateStore`;
- adapters around `NotificationPublisher`;
- SQLAlchemy Core based repositories when a bot module needs module-owned persistence;
- logging and metrics wrappers.

Forbidden:

- Bybit private order clients;
- Binance candle sync;
- standalone bot DB schema creation;
- direct exchange order placement;
- direct legacy runtime startup;
- legacy `psycopg2`, `asyncpg`, or raw DB helper access from root-level bots.

## Persistence And SQLAlchemy Core

Platform-native trading bots must use the same SQLAlchemy Core persistence direction as
the implemented Python services in this repository.

Rules:

- Use SQLAlchemy Core table definitions and repositories for bot-owned persistence when
  the module needs data beyond the generic platform `StateStore`.
- Do not use classic SQLAlchemy ORM for Bot Platform persistence unless the service-wide
  standards are changed first.
- Keep SQLAlchemy Core table definitions and repositories in the module `infrastructure/`
  or in shared Bot Platform persistence modules, not in `domain/`.
- Keep transaction ownership in Bot Platform application services or repository
  boundaries.
- Use Alembic migrations for schema changes.
- Keep monetary, price, quantity, volume, and confidence values as `Decimal` in Python and
  `NUMERIC` in PostgreSQL.
- Do not reuse legacy bot DB helpers, standalone table creation scripts, or direct
  `psycopg2` access.
- Do not let strategy/domain code import SQLAlchemy engines, tables, rows, or
  repositories directly.
- Prefer platform `StateStore` for simple per-instance runtime state; introduce
  module-owned SQLAlchemy Core tables only when the state has query/reporting requirements
  that do not fit key-value runtime state.

## What To Move From Legacy Bots

Move only platform-safe strategy code.

Good candidates:

- domain models that describe candles, signals, positions, target decisions, and strategy state;
- pure signal generation and planner logic;
- portfolio/risk rules that do not call external systems;
- strategy config defaults that are safe in platform mode;
- tests for pure strategy behavior;
- adapter mapping tests.

Do not move:

- root-level `main.py` CLI entrypoints;
- Dockerfiles for standalone bots;
- `.env`, `.venv`, caches, or generated files;
- direct Bybit private clients;
- live order executors;
- Binance sync jobs;
- standalone DB migration helpers;
- Telegram implementation details;
- scheduler code that owns a process loop outside Bot Platform workers.

If a legacy component mixes pure decisions with side effects, split it before moving:

- pure decision logic goes into `trading_bots/<module>/domain/`;
- orchestration goes into `trading_bots/<module>/application/`;
- platform adapters go into `trading_bots/<module>/infrastructure/`;
- live exchange execution remains outside Bot Platform.

## Registry And Discovery

The registry should support module discovery from:

```text
bot_platform_service.trading_bots
```

Target adapter paths:

```text
bot_platform_service.trading_bots.spot_grid.adapter
bot_platform_service.trading_bots.spot_greenwich.adapter
```

Registry validation should accept the new prefix:

```text
bot_platform_service.trading_bots.
```

The old prefix may remain temporarily for compatibility:

```text
bot_platform_service.infrastructure.bot_modules.
```

Long-term, `bot_platform_service.infrastructure.bot_modules` should be considered deprecated after all migrated modules pass readiness gates.

Discovery rules:

- scan only direct child packages under `trading_bots`;
- import `manifest.py` only during metadata discovery;
- import or read `config_schema.py` only through the metadata/config-schema discovery path;
- do not import `adapter.py` during manifest validation;
- reject duplicate `module_id`;
- reject module IDs ending in `_bot`;
- persist `module_id`, `display_name`, `version`, `adapter_path`, `adapter_class`,
  manifest JSON, and config schema JSON in platform metadata storage.

Required platform data-model changes:

- extend the module metadata domain model with `adapter_class` or a single fully-qualified
  adapter import path;
- extend the module repository contract to persist config schema JSON;
- extend `_bot_platform.bot_modules` or add a related metadata table for config schema
  JSON and schema version;
- add Alembic migrations for the metadata shape;
- update registry tests to cover manifest and config schema persistence.

## Resolver Design

The production resolver should map persisted module metadata to adapter instances.

Required behavior:

- load active module metadata from `_bot_platform.bot_modules`;
- import `adapter_path` only when resolving a module for validation or execution;
- load the adapter class from persisted metadata, not from a hardcoded mapping;
- construct a fresh adapter or safe reusable adapter per instance run;
- fail with a redacted error if the adapter path is missing or invalid;
- never import root-level legacy bot packages for migrated modules;
- do not branch on concrete module IDs such as `if module_id == "spot_grid"`;
- do not keep a hardcoded `SUPPORTED_BOTS` list for runtime resolution.

Illustrative persisted mapping:

```text
module_id=spot_grid
adapter_path=bot_platform_service.trading_bots.spot_grid.adapter
adapter_class=SpotGridAdapter

module_id=spot_greenwich
adapter_path=bot_platform_service.trading_bots.spot_greenwich.adapter
adapter_class=SpotGreenwichAdapter
```

This mapping must be data-driven from registry metadata. It must not be represented as a
runtime constant in platform code.

## Admin Auto-Discovery And Config UI

New trading bot modules should appear automatically in the admin cabinet after they are
added under `src/bot_platform_service/trading_bots/` and discovered by Bot Platform.

The admin cabinet must not import Python modules directly. It should receive bot metadata
and configuration schemas through backend APIs.

Admin UI and admin API behavior must be schema-driven. They must not use bot-specific
branches such as `if moduleId === "spot_grid"` for normal module listing, instance
creation, or config form rendering.

### Metadata Source

Each bot module must expose:

- `manifest.py` for module identity, display metadata, supported modes, required
  timeframes, and required market data;
- `config_schema.py` for machine-readable configuration fields and validation metadata;
- `adapter.py` for runtime execution only, not for admin discovery.

`config_schema.py` should expose a JSON-safe schema object. Example:

```python
CONFIG_SCHEMA = {
    "schema_version": 1,
    "sections": [
        {
            "key": "market_data",
            "label": "Market Data",
            "fields": [
                {
                    "key": "symbols",
                    "type": "symbol_list",
                    "label": "Symbols",
                    "required": True,
                    "default": ["ETHUSDT"],
                },
                {
                    "key": "timeframes",
                    "type": "timeframe_list",
                    "label": "Timeframes",
                    "required": True,
                    "default": ["1h", "4h"],
                    "allowed": ["1h", "4h", "1d"],
                },
            ],
        }
    ],
}
```

Rules:

- schema content must be JSON-safe;
- schema content must not include secrets;
- secret fields must use references, for example `secret_ref`;
- labels and descriptions are admin UI metadata, not validation authority;
- backend validation remains authoritative before saving an instance config.
- `config_schema.py` must be lightweight-import safe: it must not import strategy runtime,
  SQLAlchemy, exchange clients, DB clients, network SDKs, or root-level legacy bot
  packages.

### Supported Admin Field Types

Admin UI should support a stable set of field types before bot-specific custom widgets are
introduced:

- `string`
- `integer`
- `decimal`
- `boolean`
- `enum`
- `symbol`
- `symbol_list`
- `timeframe`
- `timeframe_list`
- `secret_ref`
- `object`
- `array`

Each field may define:

- `key`;
- `type`;
- `label`;
- `description`;
- `required`;
- `default`;
- `allowed`;
- `min`;
- `max`;
- `item_schema` for arrays;
- `fields` for nested objects.

### Backend API Boundary

The identity/admin backend should expose Bot Platform metadata through admin endpoints,
but it should not become the owner of Bot Platform metadata. Bot Platform remains the
source of truth for bot modules, config schemas, instances, runs, health, and signals.

Recommended service boundary:

- Bot Platform owns internal platform APIs or application services for module metadata,
  instance lifecycle, validation, runs, health, and signals.
- Identity/admin API exposes operator-facing routes and delegates to Bot Platform through
  an internal API, service client, or explicit application boundary.
- The admin frontend talks to the identity/admin API, not directly to Python modules or
  platform database tables.

Exact routing can be adjusted to the final service boundaries, but the required
operator-facing capabilities are:

```text
GET    /admin/bot-modules
GET    /admin/bot-modules/:moduleId
GET    /admin/bot-modules/:moduleId/config-schema
POST   /admin/bot-instances
GET    /admin/bot-instances
GET    /admin/bot-instances/:instanceId
PATCH  /admin/bot-instances/:instanceId/config
POST   /admin/bot-instances/:instanceId/validate
POST   /admin/bot-instances/:instanceId/enable
POST   /admin/bot-instances/:instanceId/disable
```

The API should return only persisted/discovered metadata. It should not dynamically import
strategy code per frontend request.

### Admin UI Behavior

The admin cabinet should:

- list active discovered bot modules;
- show module details from `manifest.py`;
- render configuration forms from `config_schema.py`;
- allow creating a bot instance for a module;
- allow editing instance config with schema-aware validation;
- show validation errors returned by backend module validation;
- show instance status, mode, symbols, timeframes, latest health, latest run, and latest
  signals where available;
- avoid a raw JSON editor as the primary workflow, though an advanced JSON view may be
  useful for operators.

Forbidden admin patterns:

- hardcoded bot lists in frontend code;
- bot-specific config forms as the primary workflow;
- module-specific route branches for ordinary module management;
- duplicated config field definitions outside `config_schema.py`;
- frontend validation that is stricter or semantically different from backend schema
  validation.

### Discovery Flow

Recommended flow:

1. Bot Platform scans `bot_platform_service.trading_bots`.
2. For each child package, it reads `manifest.py` and `config_schema.py`.
3. It validates module ID, adapter path, supported modes, required timeframes, and config
   schema shape.
4. It persists module metadata and config schema JSON.
5. Admin API exposes persisted metadata to the admin cabinet.
6. Admin UI renders module cards and config forms automatically.
7. On save, backend validates submitted config through platform validation and the module
   `validate_config` path.
8. Only validated configs can move toward enabled/running states.

### Admin Auto-Discovery Acceptance Criteria

This capability is complete when:

- adding a new package under `trading_bots/<module_name>/` with `manifest.py`,
  `config_schema.py`, and `adapter.py` is enough for the module to appear in admin after
  discovery runs;
- admin module list shows the module without frontend code changes;
- admin config form renders from schema without frontend code changes;
- backend rejects invalid config before creating or updating a bot instance;
- secrets are configured as references and never exposed as raw values;
- module metadata and config schema are cached/persisted so admin requests do not import
  strategy code directly;
- adding a new module does not require adding module-specific branches to registry,
  resolver, admin API, or admin UI code.

## Compatibility Mapping

During migration, the platform may need to compare old and new identifiers.

Recommended mapping:

```text
spot_grid_bot -> spot_grid
spot_greenwich_bot -> spot_greenwich
```

Use this mapping only for:

- migration rollout reports;
- legacy comparison tooling;
- audit context;
- rollback documentation.

Do not create new production instances with `_bot` suffix IDs.

## Signal-Only Policy

All platform-native trading bots are signal-only.

They may:

- analyze immutable market snapshots;
- produce standardized `BotSignal` values;
- publish signals in `signal_only`;
- publish notifications in `notification_only`;
- read and write runtime state through platform `StateStore`.

They must not:

- open positions;
- create, cancel, or amend exchange orders;
- manage fills;
- construct private exchange clients;
- run candle sync;
- run backfill jobs;
- own a standalone scheduler process;
- read root-level bot `.env` files.

## Market Data Contract

Trading bots must consume platform market data only through:

- `BotRunRequest.market_data`;
- `BotRuntimeContext.market_data`;
- `BotMarketSnapshot`;
- `BotMarketDataContext`.

For multi-timeframe modules:

- `spot_grid` requires `1h` and `4h`;
- `spot_greenwich` requires `1d` and `4h`.

Adapters should validate that required snapshots are present before running strategy logic.

## State Contract

Runtime state must be instance-scoped.

Rules:

- state namespace should be module-specific, for example `spot_grid.runtime`;
- state keys should include normalized symbol where symbol-specific;
- simple state writes should be represented as `BotRunResult.state_changes`;
- `BotRunOrchestrationService` or a dedicated platform state applier should persist
  `state_changes` through `StateStore` after a successful run result;
- adapters should not both call `StateStore.save()` and return the same value in
  `state_changes`;
- direct `StateStore.save()` from an adapter is allowed only for explicitly documented
  immediate side effects that cannot be represented as result state changes;
- state payloads must be JSON-safe;
- state changes must be returned in `BotRunResult.state_changes`;
- root-level bot DB tables must not be read by platform-native modules.

## Implementation Phases

Implement the migration in small phases. Each phase should leave the repository in a
working state and should not require bot-specific hardcoding in registry, resolver, admin
API, or admin UI.

### Phase 1. Align Standards And Naming

Goal: make the target structure official before code moves.

Scope:

- update Bot Platform standards to include `src/bot_platform_service/trading_bots/`;
- document `adapter.py` as the package-level adapter entrypoint exception;
- keep internal infrastructure adapter files on the `*_adapter.py` convention;
- document SQLAlchemy Core as the persistence standard for bot-owned tables;
- document that new module IDs must not end with `_bot`.

Exit criteria:

- standards and module migration plan describe the same package layout;
- no runtime code depends on the new layout yet.

### Phase 2. Add Trading Bots Package Shell

Goal: introduce the empty canonical module location without migrating strategy code.

Scope:

- add `bot_platform_service/trading_bots/__init__.py`;
- add package-level comments or docs describing the plugin-style boundary;
- do not move root-level bots yet;
- do not change existing adapter behavior yet.

Exit criteria:

- package imports cleanly;
- existing Bot Platform tests still pass.

### Phase 3. Define Config Schema Contract

Goal: make config schemas machine-readable before admin work starts.

Scope:

- define the allowed config schema shape and field types;
- add validation helpers for schema JSON safety;
- add tests for valid and invalid config schema examples;
- keep schema validation independent from concrete bot strategy imports.

Exit criteria:

- a fixture `CONFIG_SCHEMA` can be validated without importing `adapter.py`;
- schema validation rejects non-JSON-safe values and unsupported field types.

### Phase 4. Extend Registry Metadata Model

Goal: persist enough metadata for discovery, admin UI, and resolver.

Scope:

- extend module metadata with `adapter_class` or a single fully-qualified adapter import
  path;
- persist config schema JSON and config schema version;
- add Alembic migration for metadata changes;
- update repository contracts and tests.

Exit criteria:

- module metadata can store manifest JSON and config schema JSON;
- existing module registration remains backward-compatible where required.

### Phase 5. Add Trading Bot Discovery

Goal: discover modules from package metadata without executing strategy code.

Scope:

- scan direct child packages under `bot_platform_service.trading_bots`;
- import only `manifest.py` and `config_schema.py`;
- reject duplicate module IDs and `_bot` suffix IDs;
- register discovered metadata through registry;
- keep old `infrastructure.bot_modules` registration path temporarily supported.

Exit criteria:

- a fake module package can be discovered and registered;
- discovery does not import `adapter.py`;
- discovery has no hardcoded bot list.

### Phase 6. Add Data-Driven Resolver

Goal: resolve runtime adapters from persisted metadata.

Scope:

- load active module metadata from registry storage;
- import adapter module from persisted `adapter_path`;
- load adapter class from persisted `adapter_class`;
- instantiate adapters without branching on concrete module IDs;
- return redacted errors for missing or invalid adapters.

Exit criteria:

- resolver can load a fixture adapter from metadata;
- resolver tests prove no `SUPPORTED_BOTS` runtime list is required;
- existing orchestration can receive the resolver through its current protocol.

### Phase 7. Add State Change Applier

Goal: make state persistence ownership explicit before migrated bots return state changes.

Scope:

- add or wire a platform service that applies `BotRunResult.state_changes`;
- persist simple runtime state through `StateStore` after successful run results;
- define behavior for failed or partial runs;
- prevent duplicate writes when adapters call `StateStore.save()` directly.

Exit criteria:

- orchestration persists returned state changes exactly once;
- tests cover success, failure, and duplicate state-change behavior.

### Phase 8. Add Admin Metadata API Boundary

Goal: expose discovered modules and schemas to the admin backend without frontend-specific
logic.

Scope:

- expose module list, module detail, and config schema capabilities from Bot Platform;
- define the identity/admin API delegation boundary;
- return persisted metadata only;
- do not import strategy code per admin request.

Exit criteria:

- admin-facing API tests can list discovered fixture modules;
- config schema endpoint returns persisted schema JSON;
- API has no bot-specific branches.

### Phase 9. Add Generic Admin Config UI

Goal: render bot config forms from schema without bot-specific frontend code.

Scope:

- implement generic field rendering for the supported field types;
- render sections and nested fields from `config_schema.py`;
- submit config payloads to backend validation;
- show backend validation errors.

Exit criteria:

- fixture schemas render without custom components;
- adding a fixture module does not require frontend code changes;
- raw JSON editor is not the primary workflow.

### Phase 10. Create `spot_grid` Platform-Native Skeleton

Goal: add the first real module package without moving all strategy code at once.

Scope:

- create `trading_bots/spot_grid/`;
- add `manifest.py`, `config_schema.py`, and empty/fixture-safe `adapter.py`;
- keep root-level `spot_grid_bot` untouched;
- add manifest, config schema, and discovery tests for the new module metadata.

Exit criteria:

- `spot_grid` appears through discovery;
- admin metadata can show `spot_grid`;
- runtime adapter is loadable but may still return a controlled not-implemented result.

### Phase 11. Migrate `spot_grid` Domain And Application Logic

Goal: move platform-safe grid strategy logic into the new module.

Scope:

- move or copy pure domain logic into `trading_bots/spot_grid/domain/`;
- move platform one-run orchestration into `trading_bots/spot_grid/application/`;
- replace legacy imports with package-local imports;
- keep exchange, sync, DB helpers, scheduler, and CLI out of the module.

Exit criteria:

- domain tests pass against the platform-native package;
- source-boundary tests prove no private exchange or sync imports exist.

### Phase 12. Complete `spot_grid` Adapter

Goal: make `spot_grid` runnable in platform modes.

Scope:

- map platform market snapshots into module-local application ports;
- map grid target decisions into deterministic `BotSignal` records;
- return diagnostics and state changes through `BotRunResult`;
- support `dry_run`, `notification_only`, and `signal_only` as applicable.

Exit criteria:

- adapter contract tests pass;
- duplicate snapshot tests preserve signal idempotency;
- no order placement or candle sync path exists in platform mode.

### Phase 13. Create `spot_greenwich` Platform-Native Skeleton

Goal: add the second real module package with the same top-level template.

Scope:

- create `trading_bots/spot_greenwich/`;
- add `manifest.py`, `config_schema.py`, and empty/fixture-safe `adapter.py`;
- keep root-level `spot-greenwich-bot` untouched;
- add manifest, config schema, and discovery tests for the new module metadata.

Exit criteria:

- `spot_greenwich` appears through discovery;
- admin metadata can show `spot_greenwich`;
- runtime adapter is loadable but may still return a controlled not-implemented result.

### Phase 14. Migrate `spot_greenwich` Domain And Application Logic

Goal: move platform-safe Greenwich strategy logic into the new module.

Scope:

- move or copy pure models, signals, planner, and execution policy into
  `trading_bots/spot_greenwich/domain/`;
- move platform one-run orchestration into `trading_bots/spot_greenwich/application/`;
- replace legacy imports with package-local imports;
- keep exchange, sync, migrations, scheduler, command dispatch, and CLI out of the module.

Exit criteria:

- domain tests pass against the platform-native package;
- source-boundary tests prove no private exchange or sync imports exist.

### Phase 15. Complete `spot_greenwich` Adapter

Goal: make `spot_greenwich` runnable in platform modes.

Scope:

- map platform D1 and 4H snapshots into module-local application ports;
- map Greenwich execution decisions into deterministic `BotSignal` records;
- return notifications, diagnostics, and state changes through `BotRunResult`;
- support `dry_run`, `notification_only`, and `signal_only` as applicable.

Exit criteria:

- adapter contract tests pass;
- duplicate snapshot tests preserve signal idempotency;
- no order placement or candle sync path exists in platform mode.

### Phase 16. Wire Orchestration End-To-End

Goal: run enabled platform-native module instances from scheduler or market-data events.

Scope:

- use discovered registry metadata for eligible instances;
- use data-driven resolver for adapters;
- pass `BotMarketDataContext` into runs;
- persist runs, run events, signals, notifications, diagnostics, and state changes.

Exit criteria:

- `BotRunOrchestrationService` can run fixture, `spot_grid`, and `spot_greenwich`
  instances without hardcoded branches;
- failed instance runs do not stop unrelated instances.

### Phase 17. Add Migration Compatibility And Rollout Checks

Goal: keep legacy root-level bots as rollback while new modules are validated.

Scope:

- add compatibility mapping only for reports and rollout tooling;
- compare legacy outputs with platform-native outputs;
- track stable runs for `dry_run`, `notification_only`, and `signal_only`;
- keep production instance creation on suffix-free module IDs.

Exit criteria:

- compatibility mapping is not used by resolver or admin UI;
- rollout gates can be evaluated for both modules.

### Phase 18. Finalize Documentation And Retire Old Adapter Path

Goal: make the new module framework the documented standard and remove compatibility
code from `bot_platform_service`.

Scope:

- finalize `STANDARDS.md` after the migrated modules and runtime path are proven;
- finalize `how_to_add_new_bot_module.md`;
- finalize `migration_rollout.md`;
- finalize `production_readiness.md`;
- remove `bot_platform_service.infrastructure.bot_modules` after both platform-native
  modules pass readiness gates and production resolver traffic no longer depends on the
  legacy adapter path;
- remove legacy adapter exports, tests, bootstrap wiring, compatibility resolver paths,
  and documentation references that exist only for `infrastructure.bot_modules`;
- perform a `bot_platform_service` legacy cleanup pass to remove unused compatibility
  code, old `_bot` module ID paths, temporary mapping helpers, and any remaining
  references to root-level bot runtime imports;
- keep root-level standalone bots only if they are still needed as external rollback
  references; they must not be imported by `bot_platform_service` after cleanup.

Exit criteria:

- new docs point developers to `trading_bots/`;
- `bot_platform_service.infrastructure.bot_modules` no longer exists in service runtime
  code;
- resolver, discovery, admin metadata API, admin UI, orchestration, and tests use
  `bot_platform_service.trading_bots.*.adapter`;
- no `bot_platform_service` code imports root-level legacy bot packages;
- old adapter path is documented only in migration history or rollback notes, not as an
  available extension path.

## Testing Requirements

Each migrated bot must have:

- manifest validation tests;
- config schema validation tests;
- `BotModuleContractHarness` tests;
- adapter `dry_run` tests;
- adapter `notification_only` tests where notifications are supported;
- adapter `signal_only` idempotency tests;
- domain tests for strategy decisions;
- source-boundary smoke tests proving no private exchange clients or sync jobs are imported;
- registry discovery tests for the new `trading_bots` prefix;
- admin metadata/config-schema API tests;
- admin form schema fixture tests for supported field types;
- resolver tests for `module_id -> adapter` loading.

Existing root-level bot tests should remain untouched until the platform-native tests cover the migrated behavior.

## Documentation Updates Covered By Phases

The implementation phases above include documentation updates. The documents that must be
kept aligned are:

- `bot-platform-service/STANDARDS.md`;
- `bot-platform-service/docs/how_to_add_new_bot_module.md`;
- `bot-platform-service/docs/migration_rollout.md`;
- `bot-platform-service/docs/production_readiness.md`.

Those documents currently describe or mention adapter paths under `infrastructure/bot_modules`.
Phase 1 should align the target standards before code moves; Phase 18 should finalize the
documentation after migrated modules and runtime wiring are proven.

## Acceptance Criteria

The migration design is complete when:

- `src/bot_platform_service/trading_bots/` is the canonical location for trading modules;
- `spot_grid` and `spot_greenwich` exist as platform-native packages;
- neither platform-native module imports root-level legacy bot packages;
- both modules expose `manifest.py`, `adapter.py`, and `config_schema.py`;
- registry discovery can register both modules from manifests;
- registry discovery can persist both module config schemas;
- admin APIs can expose both module manifests and config schemas;
- admin UI can render configurable forms for both modules without bot-specific frontend code;
- production resolver can instantiate both adapters by module ID;
- `BotRunOrchestrationService` can run enabled instances for both modules;
- resolver and admin UI are schema-driven and do not hardcode concrete module IDs for
  normal operation;
- all platform runs remain signal-only;
- `bot_platform_service.infrastructure.bot_modules` has been removed after readiness
  gates pass;
- `bot_platform_service` legacy compatibility code has been cleaned up;
- root-level bots remain available as rollback references only outside
  `bot_platform_service` until explicitly archived or deleted.

## Open Decisions

- Whether to keep temporary compatibility rows for `spot_grid_bot` and `spot_greenwich_bot` in `_bot_platform.bot_modules`.
- Whether migrated modules should share common strategy primitives in `trading_bots/common/`.
- Whether module-local tests should live only under `bot-platform-service/tests/` or also inside each module package.
- Whether adapter instances should be created per run or cached per module.
- Whether root-level legacy bots outside `bot_platform_service` will be archived, deleted,
  or retained after readiness gates pass.
