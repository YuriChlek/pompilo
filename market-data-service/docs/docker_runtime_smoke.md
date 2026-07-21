# Market Data Docker Runtime Smoke

This smoke verifies the long-running `market_data` Docker runtime path:

- PostgreSQL and Redis are healthy;
- `market_data_migrate` applies Alembic migrations;
- bootstrap `collect --once` creates candles and snapshots;
- bootstrap range does not create outbox or Redis Stream events;
- incremental/live `collect --once` creates published outbox events and Redis Stream messages;
- `outbox:cleanup` removes old terminal outbox records without deleting candles;
- an idempotent rerun does not create duplicate candles or extra Redis Stream messages;
- `market_data` stays running and exposes `/health/ready` and `/metrics`.

Run from the repository root:

```bash
bash market-data-service/scripts/docker_runtime_smoke.sh
```

Useful overrides:

```bash
PROJECT_NAME=pampilo-platform \
ENV_FILE=control-panel/.env \
SMOKE_WAIT_SECONDS=120 \
bash market-data-service/scripts/docker_runtime_smoke.sh
```

The smoke intentionally uses `MARKET_DATA_PROVIDER_MODE=fixture` and the isolated
`market-data-events-smoke` Redis Stream by default. It resets `_market_data` smoke
state inside the configured database, so do not point it at shared production data.
It does not use or inspect legacy bot directories.

By default, the smoke does not include `docker-compose.dev.yaml`, so it does not
bind PostgreSQL, Redis, or HTTP ports on the host. To include local dev port
bindings explicitly:

```bash
SMOKE_INCLUDE_DEV_COMPOSE=true bash market-data-service/scripts/docker_runtime_smoke.sh
```

If it fails with zero `market_candles`, `market_snapshots`, published outbox events,
or Redis Stream messages, the runtime did not complete the expected bootstrap or
incremental provider sync. Inspect:

```bash
docker compose -p pampilo-platform \
  --env-file control-panel/.env \
  -f infra/compose/docker-compose.yaml \
  -f infra/compose/docker-compose.dev.yaml \
  logs --tail=200 market_data
```
