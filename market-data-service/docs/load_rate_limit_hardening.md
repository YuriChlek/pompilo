# Market Data Service Load And Rate-Limit Hardening

## Load Shape

The service validates scheduler load by generating sync jobs for the configured symbol universe across `1h`, `4h`, and `1d`.
Load tests must assert one idempotent job per `source + provider_symbol + timeframe + close_time`.

## Concurrency Limits

Provider calls are guarded by `AsyncConcurrencyLimiter`.
The Binance provider defaults to `BINANCE_MAX_CONCURRENT_REQUESTS=8`.

## Rate Limits

Binance `429` responses are classified as retryable provider errors and recorded with:

- `market_data_provider_errors_total`
- `market_data_provider_rate_limited_total`

Rate-limit errors must not abort unrelated symbol/timeframe jobs in the same scheduler cycle.

## Circuit Breaker

The provider circuit breaker opens after `BINANCE_CIRCUIT_BREAKER_FAILURE_THRESHOLD` consecutive provider failures.
It moves to half-open after `BINANCE_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS` and closes after the next successful call.

## Lag Checks

Outbox lag is measured with `market_data_outbox_lag_seconds`.
Queue lag is measured with `market_data_queue_lag_seconds`.
It should be checked together with outbox lag during production-scale smoke runs.
