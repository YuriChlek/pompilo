# PostgreSQL Outage Runbook

## Symptoms

- `/health/readiness` fails or API returns `503`.
- Security event writes fail with critical policy.
- Mail outbox lag increases.

## Immediate Checks

```bash
docker compose -p pampilo-platform --env-file register-service/.env \
  -f infra/compose/docker-compose.yaml -f infra/compose/docker-compose.dev.yaml \
  --profile infra ps postgres
docker logs pampilo-platform-postgres-1 --tail 200
```

## Recovery

1. Confirm disk space and container health.
2. If PostgreSQL is down, restart only `postgres`.
3. Re-run `/health/readiness`.
4. Run `register-service/ops/scripts/backup-restore-smoke.sh` after recovery.
5. Check `mail_outbox_lag_seconds` and outbox relay logs.

## Data Safety

Never run `down -v` during incident response. Take a logical dump before destructive maintenance.
