# Production Readiness

This document fixes Bot Platform as the standard runtime for Python trading bots.

Bot Platform remains signal-only in production. It must not open positions, create exchange orders, construct private exchange clients, or run legacy candle sync jobs. Standalone CLI flows remain available as rollback until each module has passed the readiness gates below.

## Scope

Production readiness applies to:

- platform-native `spot_grid` through `bot_platform_service.trading_bots.spot_grid.adapter`;
- platform-native `spot_greenwich` through `bot_platform_service.trading_bots.spot_greenwich.adapter`;
- standalone `spot_grid_bot` and `spot_greenwich_bot` only as external rollback and
  comparison references;
- any new Python bot module added through `trading_bots/<module_id>/`, manifest,
  config schema, adapter, and contract tests.

The implementation directory is:

```text
bot-platform-service/
```

New production module IDs must be suffix-free. Use `spot_grid` and `spot_greenwich`, not
`spot_grid_bot` or `spot_greenwich_bot`, for platform-native instances.

## Readiness Gates

A bot module can be treated as production-ready on Bot Platform only when all gates pass:

- manifest validation passes;
- `BotModuleContractHarness` passes;
- platform adapter uses immutable market snapshots;
- no platform-mode path creates private exchange clients;
- no platform-mode path opens positions or creates exchange orders;
- signal payloads use deterministic `signal_key` and `payload_hash`;
- duplicate snapshot delivery does not create duplicate runs or duplicate signals;
- failed bot instance does not stop unrelated instances;
- `dry_run`, `notification_only`, and `signal_only` rollout gates are complete;
- health endpoint, metrics, dashboard, and alerts are configured;
- backup/restore state procedure has been tested;
- rollback plan has been tested.

## Operator Runbook

Use this order for a production incident or release check:

1. Check platform health.

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/unit/test_stage_14_observability_baseline.py
```

2. Check module contract conformance.

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/unit/test_stage_15_bot_module_contract_harness.py
```

3. Check rollout state and rollback gates.

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/unit/test_stage_17_migration_rollout.py
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/smoke/test_stage_17_migration_rollout_documentation.py
```

4. Check production readiness documentation.

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/smoke/test_stage_18_production_readiness_documentation.py
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/unit/test_stage_18_production_readiness_runtime.py
```

5. Check the full Bot Platform suite before promotion.

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests
```

## Rollback Plan

Rollback is instance-scoped and immediate:

- disable the affected Bot Platform instance;
- keep platform records for audit and diagnostics;
- do not delete `bot_runs`, `bot_run_events`, `bot_signals`, runtime state, or audit events;
- restore standalone CLI scheduling for the same module;
- compare latest platform diagnostics with standalone output;
- keep Bot Platform disabled for that instance until the root cause is fixed and tests pass again.

Standalone rollback commands are documented in `docs/migration_rollout.md`.

Rollback hard stops:

- platform mode attempts to create a private exchange client;
- platform mode attempts to open a position or create an exchange order;
- platform mode starts candle sync or backfill;
- duplicate event delivery creates duplicate runs or duplicate signals;
- one failed instance affects unrelated instances.

## Backup/Restore State

State backup must cover Bot Platform schema data only:

```text
_bot_platform.bot_modules
_bot_platform.bot_instances
_bot_platform.bot_instance_configs
_bot_platform.bot_runtime_state
_bot_platform.bot_runs
_bot_platform.bot_run_events
_bot_platform.bot_signals
_bot_platform.bot_permissions
_bot_platform.bot_secrets_refs
_bot_platform.bot_health_checks
_bot_platform.bot_audit_events
```

Backup rules:

- use database-native backups for `_bot_platform`;
- include schema migrations and Alembic revision state;
- keep secrets as references only, not raw secret values;
- verify backup freshness before enabling `signal_only`;
- store restore evidence with the release checklist.

Restore rules:

- restore into a non-production database first;
- run Alembic migrations to the expected revision;
- run repository and signal idempotency tests;
- verify runtime state optimistic locking;
- verify duplicate signal publish returns the existing signal record;
- promote restored state only after health reports are clean.

## Security Review

Security review must confirm:

- manifests do not contain secrets;
- logs, alerts, audit payloads, and diagnostics do not contain raw secret values;
- `SecretProvider` resolves only refs for the current `instance_id`;
- permissions are instance-scoped;
- `SignalPublisher`, `NotificationPublisher`, `StateStore`, and `SecretProvider` permissions are separate;
- platform adapters do not create private exchange clients;
- Bot Platform does not open positions, create orders, manage fills, or call private exchange execution APIs.

## Load Tests

Load testing must prove the platform can handle expected event pressure without losing idempotency or isolation.

Minimum scenarios:

- repeated `CandleBatchReady` delivery for the same snapshot;
- multiple active instances for the same module;
- mixed `dry_run`, `notification_only`, and `signal_only` instances;
- signal publisher duplicate delivery;
- health report generation while runs are active;
- dashboard metric recording under failed and successful runs.

Load test acceptance:

- duplicate events remain idempotent;
- no duplicate signals are created for the same `signal_key`;
- failed run count is visible in metrics;
- health status reflects failed modules, instances, and runs;
- unrelated instances keep running after one instance fails.

## Failure/Recovery Tests

Failure and recovery testing must include:

- missing snapshot;
- incomplete snapshot;
- stale snapshot;
- unsupported timeframe;
- bot adapter exception;
- signal publish retry;
- platform restart after partially completed run;
- restore of `_bot_platform` state into a clean database.

Recovery acceptance:

- failed bot instance is isolated;
- run result contains redacted error details;
- retry does not duplicate runs or signals;
- health endpoint reports unhealthy state during failure;
- alerts are emitted for failed modules, failed instances, or failed runs;
- after recovery, the platform returns to healthy status.

## Standard Runtime Policy

After readiness gates pass, Bot Platform is the standard runtime for Python trading bots.

New bots must be added through:

- manifest;
- config schema;
- package-local adapter under `trading_bots/<module_id>/adapter.py`;
- contract tests;
- snapshot-based market data;
- standardized `BotSignal` output;
- production readiness checklist update.

The retired `bot_platform_service.infrastructure.bot_modules` adapter path is not a
supported production runtime or extension path. Legacy standalone CLI mode is retained as
an external rollback path only, not as a primary production runtime, and must not be
imported by `bot_platform_service`.
