# SLOs And Alert Thresholds

## Beta SLOs

| Area | SLO |
| --- | --- |
| API readiness | 99.5% successful `/health/readiness` checks over 7 days |
| Auth latency | p95 register/login/refresh under 800 ms during beta load |
| Refresh reliability | 99.9% successful refreshes excluding invalid/revoked tokens |
| Mail enqueue | p95 outbox enqueue under 500 ms |
| Mail delivery pipeline | 99% delivery success for configured SMTP over 24 hours |
| Mail backlog | `bullmq_queue_depth` below 100 for 10 minutes |
| Outbox lag | `mail_outbox_lag_seconds` below 300 seconds for 10 minutes |
| Security audit | login/session/security events searchable in Loki within 2 minutes |

## Alert Thresholds

| Signal | Warning | Critical | First action |
| --- | --- | --- | --- |
| `/health/readiness` | 2 failures in 5 minutes | 5 failures in 5 minutes | Check Redis and DB runbooks |
| `bullmq_queue_depth` | > 100 for 10 minutes | > 500 for 10 minutes | Check Redis, workers, SMTP |
| `mail_outbox_lag_seconds` | > 300 for 10 minutes | > 900 for 10 minutes | Check outbox relay and DB locks |
| `smtp_delivery_success_rate` | < 0.98 for 15 minutes | < 0.90 for 15 minutes | Check SMTP runbook |
| `smtp_delivery_errors_total{category="auth"}` | any increase | sustained increase for 5 minutes | Rotate SMTP credentials |
| API error logs | > 10 errors in 5 minutes | > 50 errors in 5 minutes | Open Platform Operations dashboard |
| Security event write failures | any critical failure | repeated critical failures | Check PostgreSQL health immediately |

## Dashboard Checks

- Grafana folder: `Platform`
- Dashboard: `Platform Operations`
- Loki datasource uid: `platform-loki`
- Audit/security query: `{service="pampilo-api", channel=~"auth|auth-token"} | json`
- Mail alert query: `{service="pampilo-api", channel="mail"} | json | level=~"warn|error|fatal"`

## Beta Load Gate

Run the smoke load probe against staging or a local root stack:

```bash
node register-service/ops/load/register-service-smoke-load.mjs \
  --base-url=http://localhost:3000 \
  --concurrency=8 \
  --duration-seconds=120
```

Pass criteria:

- `failedRequests` is `0`;
- p95 should be checked from platform logs or gateway metrics;
- no critical security event write failures;
- `bullmq_queue_depth < 100`;
- `mail_outbox_lag_seconds < 300`.
