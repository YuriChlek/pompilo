# Register Service Release Checklist

Use this checklist before promoting the cleaned register service to staging or production.

## Pre-release Gates

- Confirm the release commit includes API, client, root compose, ops docs, migrations, and dashboards.
- Run clean installs:
  - `npm ci` in `register-service/api`
  - `npm ci` in `register-service/client`
- Run builds:
  - `npm --prefix register-service/api run build`
  - `npm --prefix register-service/api run build:cli`
  - `npm --prefix register-service/client run build`
- Run tests and quality checks:
  - `npm --prefix register-service/api run test:unit -- --runInBand`
  - `npm --prefix register-service/client run test`
  - `npm --prefix register-service/client run lint`
  - `npm --prefix register-service/client run lint:responsive`
  - `npm --prefix register-service/client run lint:cycles`
- Run compose validation:
  - `make docker-config-dev`
  - `make docker-config-prod`
- Run dependency audit gates:
  - `npm --prefix register-service/api audit --audit-level=high`
  - `npm --prefix register-service/client audit --audit-level=high`
- Verify migration-from-zero on an empty database with `npx drizzle-kit migrate`.
- Verify root compose starts the identity stack from the repository root.
- Verify `/health/readiness` returns healthy and `/metrics` returns Prometheus text.

## Functional Acceptance

- Register a generic customer user through `POST /auth/register`.
- Confirm public registration rejects unknown or privileged `role` fields.
- Confirm registration creates:
  - `users` row with `role=user`
  - `email_verifications` row
  - `mail_outbox` row
  - `identity_outbox_events` row
  - primary `memberships` row
- Verify login, refresh, logout, checkpoint, password reset, email change, sessions, known devices, and account deletion flows.
- Confirm `POST /integration/trading/onboarding-token` is blocked before email verification.
- Confirm the same endpoint returns a short-lived Bearer token after email verification.
- Confirm the token contains `aud=trading-service`, `scope=trading:onboarding`, `tenantId`, `membershipRole`, and `emailVerified=true`.

## Operational Acceptance

- Run `make ops-load-smoke` against the target environment.
- Run `make ops-backup-restore-smoke` against the target environment or staging clone.
- Confirm SLO dashboards and alerts from `register-service/ops/slo-alerts.md`.
- Confirm audit/security search works in Grafana/Loki.
- Confirm DB, Redis, SMTP, and integration outage runbooks are reachable by on-call staff.

## Release Decision

- No unresolved critical or high defects.
- No critical or high dependency advisories.
- Migration and rollback plan reviewed.
- Staging smoke completed after deployment.
- Production deployment window and owner confirmed.
