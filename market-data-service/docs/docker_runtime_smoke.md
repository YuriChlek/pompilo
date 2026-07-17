# Market Data Docker Runtime Smoke

This smoke verifies the long-running `market_data` Docker runtime path:

- PostgreSQL and Redis are healthy;
- `market_data_migrate` applies Alembic migrations;
- `market_data` stays running and exposes `/health/ready`;
- scheduler creates runtime work;
- candles, batches, snapshots, outbox events, and Redis Stream messages exist.

Run from the repository root:

```bash
bash market-data-service/scripts/docker_runtime_smoke.sh
```

Useful overrides:

```bash
PROJECT_NAME=pampilo-platform \
ENV_FILE=control-panel/.env \
MARKET_DATA_HTTP_HOST_PORT=8010 \
SMOKE_WAIT_SECONDS=120 \
bash market-data-service/scripts/docker_runtime_smoke.sh
```

The smoke intentionally does not use or inspect legacy bot directories.

If it fails with zero `market_candles`, `market_snapshots`, or Redis Stream messages, the runtime did not complete a provider sync. Inspect:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  logs --tail=200 market_data
```
