# How to Add a New Bot Module

This guide describes the standard process for adding a Python bot module to Bot Platform Service.

Bot Platform is signal-only. A bot module may analyze immutable market snapshots and return standardized `BotSignal` records. It must not open positions, create exchange orders, manage fills, construct private exchange clients, or run market-data synchronization.

## Recommended Structure

Create platform-native bot module code under:

```text
bot-platform-service/src/bot_platform_service/trading_bots/
```

For a new bot named `example`, use:

```text
bot_platform_service/trading_bots/example/
├── __init__.py
├── manifest.py
├── adapter.py
├── config_schema.py
├── domain/
├── application/
└── infrastructure/
```

The package-level `adapter.py` is the Bot Platform entrypoint. This is the only adapter
file exempt from the broader `*_adapter.py` naming convention because the package name
already scopes it. Internal adapters under `trading_bots/<module_id>/infrastructure/`
must still use descriptive `*_adapter.py` file names.

If the bot has its own standalone package, keep standalone CLI/runtime code outside Bot
Platform. Add package-local application services only when dependency injection is needed.
Platform code must accept platform-provided ports and must not construct database clients,
exchange clients, synchronizers, or standalone schedulers.

Do not add bot runtime adapters under service-level `infrastructure/`. The retired
`bot_platform_service.infrastructure.bot_modules` compatibility path is not a supported
extension point.

## Manifest

Each bot module must have manifest metadata equivalent to:

```python
RAW_MANIFEST = {
    "module_id": "example",
    "display_name": "Example",
    "version": "1.0.0",
    "supported_modes": ("dry_run", "notification_only", "signal_only"),
    "required_timeframes": ("1h", "4h"),
    "required_market_data": ("snapshots",),
    "supports_multi_symbol": True,
    "config_schema_version": 1,
}

ADAPTER_PATH = "bot_platform_service.trading_bots.example.adapter"
ADAPTER_CLASS = "ExampleAdapter"
```

Register metadata through the registry with an adapter path under:

```text
bot_platform_service.trading_bots.example.adapter
```

Rules:

- `module_id` must match the package name.
- New `module_id` values must not end with `_bot`.
- Legacy `_bot` IDs may appear only in audit history, migration history, or rollback
  documentation.
- `ADAPTER_PATH` must point to package-local `adapter.py`.
- `ADAPTER_CLASS` must name the `BotModule` implementation inside `adapter.py`.
- Manifest data must not include secrets.

The registry validates manifest shape, supported modes, timeframe aliases, market-data
requirements, adapter path, and module ID naming before registration. Discovery must not
import `adapter.py` while validating metadata.

## Config Schema

Each platform-native module must expose a lightweight `config_schema.py` with a JSON-safe
schema object. It must not import strategy runtime, SQLAlchemy, exchange clients, DB
clients, network SDKs, or root-level legacy bot packages.

Example:

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
                }
            ],
        }
    ],
}
```

Schema content must be JSON-safe and must not contain raw secrets. Secret values must be
represented as references such as `secret_ref`.

## BotModule Adapter

The adapter must implement `BotModule`:

```python
class ExampleBotAdapter:
    module_id = "example"

    async def validate_config(self, config): ...
    async def initialize(self, context): ...
    async def dry_run(self, request): ...
    async def run_once(self, request): ...
    async def start(self, request): ...
    async def stop(self, instance_id): ...
    async def health(self, instance_id): ...
```

Use `dry_run` for non-persisting planning. Use `run_once` for `notification_only` and `signal_only`. Long-running scheduling should stay in platform workers unless the module explicitly supports `start` and `stop`.

## Runtime Context

The platform passes `BotRuntimeContext` during `initialize`.

Available capabilities:

- `market_data`
- `signal_publisher`
- `state_store`
- `notification_publisher`
- `secret_provider`
- `logger`
- `metrics`
- `clock`

Do not create replacements for these capabilities inside the adapter. Permissions are enforced by runtime wrappers, so missing permission raises a typed permission error.

## Market Data

Bot modules consume `BotMarketDataContext`:

- `primary_snapshot`
- `supporting_snapshots`

Each `BotMarketSnapshot` contains immutable ordered `BotCandle` values. Use platform timeframe aliases:

- `H1` or `1H` -> `1h`
- `H4` or `4H` -> `4h`
- `D1` or `1D` -> `1d`

Adapters must read snapshots through `MarketDataSnapshotProvider`. They must not read legacy candle tables, run Binance sync, run backfills, or import market-data service internals.

## Signals

Return standardized `BotSignal` objects. Build them with `BotSignal.build(...)` so `signal_key` and `payload_hash` are deterministic.

Required signal rules:

- `signal_key` must be deterministic.
- `payload_schema` must be explicit.
- `payload_schema_version` must be positive.
- `payload_hash` must match canonical JSON payload content.
- Signal payloads must contain only JSON-safe values supported by platform canonicalization.
- Signals are trading signals, not order intents.

Use signal types:

- `entry`
- `exit`
- `rebalance`
- `hold`
- `alert`

Use side only when meaningful:

- `buy`
- `sell`

## BotRunResult

Every run must return `BotRunResult`, not a free-form dictionary.

Required identity fields must match the request:

- `run_id`
- `instance_id`
- `module_id`
- `mode`

Populate structured fields:

- `signals`
- `notifications`
- `diagnostics`
- `state_changes`
- `error_code`
- `error_message_redacted`

Do not expose secrets or raw exception messages in `error_message_redacted`.

## Modes

Default to `dry_run` when mode is missing.

Mode behavior:

- `dry_run`: calculate signals, no signal persistence side effects.
- `notification_only`: calculate signals and route notifications through `NotificationPublisher`.
- `signal_only`: publish normalized signals through `SignalPublisher`.

No Bot Platform mode may open positions or create exchange orders.

## Config, Secrets, And Permissions

Instance config is isolated per instance. Use versioned config schemas and keep secrets out of config.

Secrets must be referenced by name and resolved through `SecretProvider`.

State must be read and written only through `StateStore`.

If a module needs queryable module-owned persistence beyond `StateStore`, use SQLAlchemy Core table definitions and repositories plus Alembic migrations. Do not use classic
SQLAlchemy ORM, legacy bot DB helpers, standalone table creation scripts, or direct
`psycopg2` access. Strategy and domain code must not import SQLAlchemy engines, tables,
rows, or repositories.

Common permissions:

- `read_market_data`
- `send_notifications`
- `publish_signals`
- `read_state`
- `write_state`
- `emit_audit_events`

Signal publishing, notifications, state access, and market-data reads are separate permissions. Do not assume one permission grants another.

## Snapshot Integration

Platform adapters should request market data through:

```python
await context.market_data.get_latest_complete_snapshot(
    source="binance_spot",
    canonical_symbol="ETHUSDT",
    timeframe="1h",
)
```

For multi-timeframe bots, use `BotMarketDataContext` with primary and supporting snapshots. Do not start sync jobs from adapters.

## Idempotency

Run idempotency is based on instance, trigger, and snapshot identity.

Signal idempotency is based on deterministic `signal_key`. Duplicate publish of the same signal must return the existing `signal_id`.

Payload schema versioning is part of the signal key. Increment `payload_schema_version` when changing the semantic shape of the payload.

## No-Exchange-Client Rule

Platform-mode code must not contain or call:

- private exchange clients;
- order executors;
- `place_order`;
- `cancel_order`;
- market buy/sell order helpers;
- direct exchange HTTP client construction;
- direct DB construction for strategy logic;
- candle sync or backfill jobs.

If a legacy bot normally produces target orders or execution decisions, the platform adapter must convert those decisions to `BotSignal` and stop there.

## Contract Tests

Every new bot module must pass `BotModuleContractHarness`.

Minimal example:

```python
from bot_platform_service.testing import BotModuleContractCase, BotModuleContractHarness


def test_example_bot_contract() -> None:
    case = BotModuleContractCase(
        manifest=manifest,
        adapter_path="bot_platform_service.trading_bots.example.adapter",
        config=config,
        module=ExampleBotAdapter(),
        source_paths=(Path("bot-platform-service/src/bot_platform_service/trading_bots/example/adapter.py"),),
    )

    BotModuleContractHarness().assert_contract(case)
```

The harness validates:

- manifest metadata;
- lifecycle methods;
- market-data DTO usage;
- signal DTO structure;
- `BotRunResult` structure;
- permission scoping;
- idempotent signal publish;
- absence of direct DB, exchange, and market-data client construction terms.

## Checklist

Before submitting a new bot module:

- Create the module package under `trading_bots/<module_id>/`.
- Define `manifest.py`, `config_schema.py`, and package-local `adapter.py`.
- Use a suffix-free `module_id`.
- Implement `BotModule`.
- Use `BotRuntimeContext` capabilities only.
- Consume `BotMarketSnapshot` and `BotMarketDataContext`.
- Return `BotRunResult`.
- Build signals with `BotSignal.build`.
- Version signal payload schemas.
- Keep secrets out of config.
- Add contract tests with `BotModuleContractHarness`.
- Add boundary tests for no exchange clients, no direct DB, and no market-data sync.
- Run targeted tests for the new module.
- Run the full Bot Platform test suite.

Recommended commands:

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/unit/test_stage_15_bot_module_contract_harness.py
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests
```
