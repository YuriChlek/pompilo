# Register Service Operations

This directory contains operational checks for the identity service:

- `load/register-service-smoke-load.mjs` runs a dependency-free HTTP smoke/load probe.
- `scripts/backup-restore-smoke.sh` verifies PostgreSQL dump/restore and Redis RDB snapshot paths.
- `runbooks/` documents outage response for DB, Redis, SMTP, and integrations.
- `slo-alerts.md` defines beta SLOs, alert thresholds, and dashboard queries.

These checks are intentionally small and repeatable. They are not a replacement for a staging
load test, but they make phase gates and incident drills executable.
