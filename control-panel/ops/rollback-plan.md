# Register Service Rollback Plan

Use this plan when a release causes authentication, registration, email, or integration failures.

## Rollback Triggers

- Readiness is below the documented SLO for 10 minutes.
- Login, refresh, or registration p95 latency exceeds the alert threshold and does not recover.
- Registration or password reset cannot enqueue mail outbox records.
- Security event writes fail for critical flows.
- Trading onboarding tokens are rejected by trading service after deployment.
- Database migration introduces a blocking query, missing table, or invalid constraint.

## Immediate Actions

- Freeze further deployments.
- Capture current API/client image tags, migration version, and compose config.
- Check `/health/readiness`, `/metrics`, API logs, PostgreSQL health, Redis health, and mail outbox metrics.
- If the failure is config-only, restore the previous environment values and restart the affected services.
- If the failure is application code, redeploy the last known good API/client images.

## Database Rollback

- Prefer forward-fix migrations for already-applied production schema changes.
- If rollback requires data restore, stop writers first and restore from the latest verified backup.
- Before restore, export current affected rows for forensic review when feasible.
- After restore, run readiness, login, refresh, registration, and mail enqueue smoke checks.

## Service Rollback

- Redeploy the last known good saved version or image tag.
- Keep PostgreSQL and Redis volumes intact unless a verified restore is explicitly required.
- Restart API first, then client/gateway if needed.
- Confirm:
  - `/health/readiness` is healthy
  - `/metrics` is scrapeable
  - login and refresh work
  - registration queues a verification email
  - trading onboarding token can be issued for a verified user

## Communication

- Record start time, owner, affected services, and customer impact.
- Post status updates every 15 minutes during active incident response.
- After recovery, document root cause, rollback steps used, data impact, and prevention tasks.
