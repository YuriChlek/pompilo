# Redis Outage Runbook

## Symptoms

- `/health/readiness` returns `503`.
- Login/refresh/session revocation can fail closed.
- BullMQ queue depth stops moving.

## Immediate Checks

```bash
docker compose -p pampilo-platform --env-file register-service/.env \
  -f infra/compose/docker-compose.yaml -f infra/compose/docker-compose.dev.yaml \
  --profile infra ps redis
docker exec pampilo-platform-redis-1 redis-cli ping
docker logs pampilo-platform-redis-1 --tail 200
```

## Recovery

1. Confirm Redis memory pressure and `noeviction` state.
2. Restart Redis if it is unhealthy.
3. Verify `redis-cli ping` returns `PONG`.
4. Check `bullmq_queue_depth` and mail relay logs.
5. Confirm login/refresh/logout flows.

## Fail-Safe Expectation

Session revocation checks fail closed: if Redis cannot verify deny-list state, protected auth flows should return a service error instead of accepting a risky session.
