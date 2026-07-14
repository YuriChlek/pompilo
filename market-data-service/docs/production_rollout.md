# Market Data Service Production Rollout

## Scope

This rollout starts Market Data Service as a production data-ingestion layer only.
It does not connect bots, strategies, execution, or other consumers.
Consumer integration must happen in a separate later plan.

## Rollout Units

Roll out one small unit at a time:

- one provider symbol group across `1h`, `4h`, and `1d`; or
- one timeframe group across the configured provider symbols.

Do not expand the next unit until the current unit has passed the stability gate.

## Stability Gate

For each rollout unit verify:

- candles are created;
- batches complete;
- snapshots are created;
- outbox events are created after commit;
- outbox lag is below threshold;
- alerts are checked and no critical alerts are active;
- the configured stable period has elapsed.

Only after this gate may the service be marked ready for a separate consumer integration plan.

## Alerts To Check

- `market_data_snapshot_stale`
- `market_data_gap_detected`
- `market_data_provider_errors_high`
- `market_data_outbox_lag_high`
- `market_data_sync_jobs_stuck`

## Rollback Decision Points

Rollback the current rollout unit when any of these happen:

- no candles are created;
- no complete batches are created;
- no snapshots are created;
- no outbox events are created;
- critical alerts remain active;
- provider failures or outbox lag remain above threshold after retry windows.

Rollback means stopping Market Data Service workers for the current rollout unit.
Existing bots/readers are not switched by this plan, so runtime rollback does not require changing bot configuration.
Schema rollback is not required to stop ingestion.

## Completion Criteria

Market Data Service is ready for the later consumer integration plan when all rollout units have passed the stability gate.
The completion state is ingestion-ready, not consumer-connected.
