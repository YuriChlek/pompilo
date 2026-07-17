# Market Snapshot Contract v1

`market-data-service` exposes latest complete market snapshots through HTTP:

```text
GET /snapshots/latest?symbol=ETHUSDT&timeframe=1h&source=BINANCE_SPOT&max_age_seconds=7200
```

The contract version is `market-snapshot.v1`. Consumers must check `contract_version`
and `status` before using candles.

## Responses

- `200`: `status=ready`, includes `snapshot` metadata and immutable `candles`.
- `404`: `status=not_ready`, no complete snapshot is available yet.
- `503`: `status=stale`, latest complete snapshot exists but is older than `max_age_seconds`.
- `400`: invalid query arguments.

All decimal price and volume fields are strings. Bot Platform must not read internal
`_market_data` tables directly for runtime execution.

## Breaking Changes

Breaking response changes require a new endpoint or contract version, for example
`market-snapshot.v2`. `market-snapshot.v1` must remain available during migration
until all consumers are upgraded.
