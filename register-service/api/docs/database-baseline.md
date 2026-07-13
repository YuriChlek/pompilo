# Register Service Database Baseline

This document defines the phase 18 database baseline for a new register-service environment.

## Migration Chain

The baseline chain is `drizzle-migrations/0000_happy_azazel.sql`.

`drizzle.config.ts` intentionally points to `src/module-drizzle/schemas/index.ts` instead of a wildcard. New tables must be explicitly exported from that aggregator before they can enter the baseline or later migrations.

## Public Tables

The approved public tables are:

- `users`
- `tenants`
- `memberships`
- `user_settings`
- `email_verifications`
- `known_devices`
- `login_challenges`
- `reauth_confirmations`
- `sessions`
- `tokens`
- `security_events`
- `password_reset_challenges`
- `email_change_challenges`
- `mail_settings`
- `mail_outbox`
- `mail_audit_events`
- `identity_outbox_events`
- `data_patches`

No gym marketplace, athlete, coach, media, profile, object-storage, program, review, chat, feedback, moderation, or sport tables are part of this baseline.

## From-Zero Migration

For a new database:

```bash
npm run cli -- db:migration:run
```

For a local disposable database that may be dropped:

```bash
NODE_ENV=development npm run cli -- db:destructive-reset
```

The destructive reset verifies:

- Drizzle migration count matches `drizzle-migrations/meta/_journal.json`;
- all approved public tables exist;
- optional seeded admin users use `platformAdmin` or `superAdmin`.

## Existing Data Migration

There is no supported in-place upgrade from the old gym marketplace schema to this baseline.

For an existing production source, use a side-by-side migration:

1. Create a new database from this baseline.
2. Export only identity-compatible user data from the old source:
   - `name`
   - `email`
   - password hash, if compatible
   - active/deleted state, if trustworthy
   - email verification timestamp, if trustworthy
3. Normalize roles into `user`, `platformAdmin`, or `superAdmin`.
4. Do not copy athlete, coach, gym, profile, media, object-storage, program, review, chat, feedback, moderation, or sport rows.
5. Recreate sessions, refresh tokens, device trust, login challenges, password reset challenges, email change challenges, and outbox records as empty state unless a dedicated, reviewed migration script is created.
6. Run application-level smoke tests against the migrated database before switching traffic.

## Rollback And Recovery

Before switching an existing environment:

1. Take a physical or logical backup of the old database.
2. Keep the old application and database read-only until the new environment passes smoke tests.
3. Switch traffic only after login, refresh, logout, registration, email verification, password reset, account security, admin login, and mail settings checks pass.
4. Roll back by routing traffic back to the old read-only snapshot or a restored copy.

After traffic is switched, this baseline is forward-only. Do not edit already-applied migration files in a shared environment; create a new migration instead.
