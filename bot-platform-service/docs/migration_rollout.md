# Migration Rollout

This document defines the controlled rollout for migrated platform-native bots.

Bot Platform remains signal-only throughout rollout. It must not open positions, create exchange orders, construct private exchange clients, or replace standalone CLI flows during migration.

## Rollout Goals

- Roll out platform-native `spot_grid` in `dry_run`, `notification_only`, and `signal_only`.
- Roll out platform-native `spot_greenwich` in `dry_run`, `notification_only`, and `signal_only`.
- Compare platform results with standalone rollback references.
- Enable `signal_only` only after a stable period.
- Keep standalone bot CLI flows available as rollback.
- Never enable position opening inside Bot Platform rollout.

## Step 1. Spot Grid Dry Run

Target module:

```text
spot_grid
```

Target platform mode:

```text
dry_run
```

Rollback command:

```bash
PYTHONPATH=spot_grid_bot python -m main dry-run
```

Gate:

- platform run returns structured `BotRunResult`;
- generated signals match expected strategy decisions;
- `BotRunOrchestrationService` platform `dry_run` creates run records and run events,
  captures diagnostics, and applies returned state changes;
- `BotRunOrchestrationService` platform `dry_run` does not publish persisted signals;
- no private exchange client is created;
- no candle sync is started;
- at least 3 stable comparable runs pass.

## Zero-Persistence Preview

Use zero-persistence preview before platform `dry_run` when operators need to inspect
planning output without creating run records, state changes, or persisted signals.

The zero-persistence path is a direct adapter/contract-harness invocation with in-memory
fixtures or read-only snapshot inputs. It must not call `BotRunOrchestrationService`,
`ManualBotRunService`, `SignalPublisher`, or `StateStore`.

Platform `dry_run` is not this preview path. It is an orchestration-controlled mode and
may have persistence side effects.

## Step 2. Spot Grid Notification Only

Target module:

```text
spot_grid
```

Target platform mode:

```text
notification_only
```

Rollback:

```text
disable the platform instance
```

Gate:

- platform run returns structured `BotRunResult`;
- generated signals and diagnostics are visible for subscribed symbols;
- notifications are routed through platform `NotificationPublisher`;
- notification payloads contain notification metadata only, not secrets, private
  execution data, or full signal payloads;
- persisted signals are not published in this mode;
- no private exchange client is created;
- no candle sync is started;
- at least 3 stable comparable runs pass.

## Step 3. Spot Greenwich Notification Only

Target module:

```text
spot_greenwich
```

Target platform mode:

```text
notification_only
```

Rollback command:

```bash
PYTHONPATH=spot-greenwich-bot python -m main analyze --dry-run --notification-only
```

Gate:

- platform run returns structured `BotRunResult`;
- notifications are routed through platform `NotificationPublisher`;
- generated signals match expected execution decisions;
- no persisted signals are required;
- no private exchange client is created;
- no candle sync is started;
- at least 3 stable comparable runs pass.

## Step 4. Standalone Comparison

For each candidate run, compare:

- symbols;
- timeframe;
- snapshot close time;
- signal type;
- side;
- reason;
- deterministic payload hash;
- diagnostics relevant to the strategy.

Comparison failures block promotion to the next rollout mode. Store comparison reports outside adapter code; adapters should remain focused on signal generation.

Legacy module IDs are retained only as rollback references:

```text
spot_grid_bot -> spot_grid
spot_greenwich_bot -> spot_greenwich
```

## Step 5. Signal Only

`signal_only` can be enabled only after:

- standalone comparison passes;
- at least 10 stable comparable runs pass for the module;
- signal publisher idempotency tests pass;
- signal audit events are visible;
- operators confirm health and alerts are quiet.

`signal_only` persists standardized `BotSignal` records. It still does not open positions.
Persisted signal events use `bot_signal.persisted.v1` metadata only; the downstream
execution service remains disconnected until the dedicated execution-service rollout.

## Rollback

Rollback is immediate:

- disable the Bot Platform instance;
- keep or restore standalone CLI scheduling;
- use the standalone rollback command for the module;
- compare latest platform run diagnostics with standalone output;
- do not delete platform run history or signals.

Standalone CLI flows are intentionally kept available until production readiness is complete.

## Hard Stops

Stop rollout if any of these occur:

- platform mode attempts to construct a private exchange client;
- platform mode references exchange order creation;
- platform mode starts candle sync or backfill;
- duplicate snapshot delivery creates duplicate runs/signals;
- failed instance blocks unrelated instances;
- signal payload schema changes without version bump;
- health endpoint reports failed modules, instances, or runs.

## Verification Commands

Run targeted migration rollout tests:

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/unit/test_stage_17_migration_rollout.py
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests/smoke/test_stage_17_migration_rollout_documentation.py
```

Run the full Bot Platform suite:

```bash
spot_grid_bot/.venv/bin/python -m pytest bot-platform-service/tests
```
