# Market Data Service Observability Baseline

## Metrics

- `market_data_sync_duration_seconds`
- `market_data_sync_rows_fetched_total`
- `market_data_sync_rows_inserted_total`
- `market_data_sync_rows_skipped_duplicate_total`
- `market_data_gap_count`
- `market_data_snapshot_age_seconds`
- `market_data_batch_status_total`
- `market_data_provider_errors_total`
- `market_data_provider_rate_limited_total`
- `market_data_queue_lag_seconds`
- `market_data_outbox_lag_seconds`
- `market_data_sync_jobs_stuck_total`

## Dashboard Panels

- Sync throughput by `source`, `canonical_symbol`, `timeframe`, and `status`.
- Sync duration p50/p95/p99 by timeframe.
- Rows fetched, inserted, and duplicate-skipped by timeframe.
- Gap count by symbol/timeframe.
- Snapshot age by symbol/timeframe.
- Provider errors and rate limited responses.
- Outbox lag and publish failures.
- Sync jobs stuck in `RUNNING`.

## Alerts

- `market_data_snapshot_stale`: `market_data_snapshot_age_seconds` above the configured timeframe threshold.
- `market_data_gap_detected`: `market_data_gap_count > 0`.
- `market_data_provider_errors_high`: provider errors above threshold.
- `market_data_outbox_lag_high`: `market_data_outbox_lag_seconds` above threshold.
- `market_data_sync_jobs_stuck`: stuck `RUNNING` jobs above zero.

## Structured Log Fields

Every market-data operational log should include:

- `source`
- `canonical_symbol`
- `timeframe`
- `batch_id`
- `snapshot_id`
- `last_closed_candle_time`
- `status`
- `gap_count`
- `correlation_id`
