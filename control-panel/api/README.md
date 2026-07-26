# Gym Marketplace API

NestJS backend for the sports coaching marketplace MVP.

## Stack

- NestJS 11
- PostgreSQL via Drizzle ORM and `pg`
- Redis via `ioredis`
- BullMQ for queue infrastructure
- Socket.IO for realtime chat and presence
- Cookie-based JWT authentication
- `sharp` for media processing
- Jest unit tests

## Current Scope

- Authentication for `athlete`, `coach`, `admin`, and `superAdmin`
- Separate auth entry points for customer roles and admin roles
- Frontend-facing API areas for athlete, coach, and admin
- Profiles, profile photo uploads, and `/uploads` static hosting
- Training programs, program sharing/copies, day completion, and exercise feedback
- Chat conversations, messaging, typing, delivery state, presence, and realtime updates
- Trainer discovery, sports catalog, reviews, feedback, moderation, translation, and media assets
- Swagger documentation at `/swagger`

## Architecture Notes

- `src/app.module.ts` wires the domain modules and the global response interceptor.
- `src/main.ts` configures validation, cookies, CORS, Swagger, `/uploads` static hosting, and the Socket.IO adapter.
- Database access is implemented through the global Drizzle provider in `src/module-drizzle/providers/drizzle.provider.ts`.
- Redis is used for BullMQ and for cross-instance Socket.IO messaging.
- If Redis is unavailable, the Socket.IO adapter falls back to local in-process rooms instead of failing startup.
- Routing is split between domain modules and frontend aggregation modules. Some controllers intentionally use empty `@Controller()` decorators and receive their public prefixes through `RouterModule.register(...)`.

## Main Route Areas

- Customer auth: `POST /register`, `/login`, `/logout`, `/refresh`, `/me`
- Admin auth: `POST /admin/login`, `/admin/logout`, `/admin/refresh`, `/admin/me`
- Athlete area: `/athlete/profile`, `/athlete/programs`, `/athlete/chat`, `/athlete/reviews`, `/athlete/media`, `/athlete/feedback`
- Coach area: `/coach/chat`, `/coach/reviews`, `/coach/profile`, `/coach/programs`, `/coach/media`, `/coach/feedback`
- Shared public/domain routes: `/sports`, `/trainers`

## Authentication Ownership

- `module-customer-auth` owns non-admin roles: `athlete` and `coach`.
- `module-admin-auth` owns admin roles: `admin` and `superAdmin`.

For non-admin roles, registration, login, logout, token refresh, and current-user lookup are served by `module-customer-auth`.
For admin roles, login, logout, token refresh, and current-user lookup are served by `module-admin-auth`.

Do not add athlete or coach authentication routes to `module-admin-auth`, and do not add admin authentication routes to `module-customer-auth`.

Admin and customer authentication cookies use `path: '/'` so browser requests can reach
the backend through the Next.js `/api/*` proxy. The scopes use distinct cookie names.

Cookie `Path` is not the auth-scope security boundary. The frontend proxy must forward only
the access cookie for the target scope, refresh endpoints must receive only the matching
refresh cookie, and the backend must validate token realm and role.

## Environment Variables

Required core variables:

- `COOKIE_DOMAIN`
- `DB_HOST`
- `DB_PORT`
- `DB_USER`
- `DB_PASSWORD`
- `DB_NAME`
- `JWT_SECRET`
- `JWT_ACCESS_TOKEN_TTL`
- `JWT_REFRESH_TOKEN_TTL`
- `ENCRYPTION_KEY`

Environment variables are validated during NestJS bootstrap. Invalid or missing core
configuration stops startup with an aggregated error list. Numeric values and booleans are
normalized before they are exposed through `ConfigService`; Redis, HTTP, and mail capacity
settings retain documented development defaults when omitted.

Defaults are provided for `NODE_ENV`, `PORT`, `CLIENT_ORIGIN`, `CLIENT_PUBLIC_URL`,
`REDIS_HOST`, `REDIS_PORT`, and `REDIS_DB`.

Optional Redis settings:

- `REDIS_PASSWORD`

Capacity settings:

- `DB_POOL_MAX` controls the PostgreSQL pool per API replica. It defaults to and is capped at `30`.
- Redis is initially sized for `2GB` with `noeviction` policy in the Docker stack.
- Full capacity targets, provider-quota alignment and load/stress acceptance criteria are
  documented in [`../docs/email_capacity_limits.md`](../docs/email_capacity_limits.md).

Mail delivery provider quota and BullMQ outbound throttling:

- `MAIL_QUEUE_LIMITER_MAX` and `MAIL_QUEUE_LIMITER_DURATION` define the queue throughput window.
- The default local/development target is `100` jobs per `1000ms`, matching the provisional capacity target of up to 100 critical emails/sec or 6,000 emails/minute.
- Keep these values at or below the real SMTP provider quota for the deployed account.
- `MAIL_WORKER_CONCURRENCY` controls worker parallelism per API replica; default is `4`.
- `MAIL_RETRY_ATTEMPTS`, `MAIL_RETRY_BACKOFF_TYPE`, `MAIL_RETRY_BACKOFF_DELAY`, and `MAIL_RETRY_BACKOFF_JITTER` configure retries. Attempts default to `3`; jitter is a `0..1` ratio and defaults to `0.2`.

## Database and Migrations

- Drizzle config lives in `drizzle.config.ts`.
- Migration files live in `drizzle-migrations/`.
- Database scripts:
  - `npm run cli -- db:migration:generate`
  - `npm run cli -- db:migration:generate --name=add-user-avatar`
  - `npm run cli -- db:migration:run`
  - `npm run cli -- db:dev:migration:run`
  - `npm run cli -- db:studio`
- Migration snapshots are kept for every journal entry. When generating a completely fresh
  baseline, the project generator also appends the PostgreSQL trigger/function and `mail_outbox`
  autovacuum settings that Drizzle schema snapshots cannot represent.

## Commands

Run these commands from the `api/` directory.

Install dependencies:

```bash
npm install
```

Application lifecycle:

| Command | Description |
|---|---|
| `npm run build` | Builds the NestJS application into `dist/`. |
| `npm run start` | Starts the NestJS application once through the Nest CLI. |
| `npm run start:dev` | Starts the API in development mode with file watching enabled. |
| `npm run start:debug` | Starts the API with the Node debugger and file watching enabled. |
| `npm run start:prod` | Runs the compiled production entrypoint from `dist/src/main` with `NODE_ENV=production`. Run `npm run build` first. |

Code quality:

| Command | Description |
|---|---|
| `npm run format` | Formats `src/**/*.{ts,tsx}` and `test/**/*.{ts,tsx}` with Prettier. |
| `npm run lint` | Runs ESLint over source, test, and script files with `--fix`. |

Database and seed scripts:

| Command | Description |
|---|---|
| `npm run cli -- db:migration:generate [--name=<migration-name>]` | Generates a Drizzle migration from the current schema. Without `--name`, drizzle-kit uses its default generated migration name. For a fresh baseline, the generator also appends required raw SQL for the `mail_outbox` notification trigger/function and autovacuum settings. |
| `npm run cli -- db:migration:run` | Applies pending Drizzle migrations using the current environment. |
| `npm run cli -- db:dev:migration:run` | Applies pending Drizzle migrations with `NODE_ENV=development`. |
| `npm run cli -- db:studio` | Opens Drizzle Studio for database inspection. |
| `npm run cli -- db:destructive-reset` | Uses `DB_NAME` from the environment, drops and recreates the `public` and `drizzle` schemas, runs migrations, seeds default sports, and optionally seeds an admin user when all admin seed env vars are provided. Refuses production and protected databases. |
| `npm run cli -- admin:create --admin-email=<email> --admin-password=<password> --admin-firstname=<firstname> --admin-lastname=<lastname> [--role=<role>]` | Creates an `admin` or `superAdmin` user. The default role is `admin`. |

## Operational CLI

The canonical operational command form is:

```bash
npm run cli -- <command> [--key=value]
```

In this command, the `--` after `npm run cli` is the npm argument separator. It is not passed to the CLI command. Command-specific arguments use `--key=value`. The only framework-level exception is command help, which uses `help <command>`.

Canonical commands:

| Command | Description |
|---|---|
| `npm run cli -- help` | Lists available CLI commands. |
| `npm run cli -- help <command>` | Shows usage for one command. |
| `npm run cli -- data-patches:push` | Runs pending data patches. |
| `npm run cli -- data-patches:list` | Lists data patches and their status. |
| `npm run cli -- data-patches:dry-run` | Shows pending data patches without applying them. |
| `npm run cli -- data-patches:create --name=patch-name` | Creates a data patch template. |
| `npm run cli -- data-patches:build` | Compiles data patches and generates the manifest. |
| `npm run cli -- db:migration:generate` | Generates a Drizzle migration with the default generated migration name. |
| `npm run cli -- db:migration:generate --name=add-user-avatar` | Generates a Drizzle migration. |
| `npm run cli -- db:migration:run` | Applies pending Drizzle migrations. |
| `npm run cli -- db:dev:migration:run` | Applies pending Drizzle migrations with `NODE_ENV=development`. |
| `npm run cli -- db:studio` | Starts Drizzle Studio. |
| `npm run cli -- db:destructive-reset` | Resets the configured development database, runs migrations, and seeds data. |
| `npm run cli -- admin:create --admin-email=admin@example.com --admin-password=Seed1234 --admin-firstname=Admin --admin-lastname=Name --role=superAdmin` | Creates an admin user. |
| `npm run cli -- demo-data:push` | Seeds demo data for development. |

Operational npm wrappers have been removed. New operational commands must be added as CLI command classes, registered in the CLI registry, and documented through the `npm run cli -- <command>` entrypoint.

Deployment steps should use `npm run build`, `npm run build:cli`, `npm run cli -- db:migration:run`, and `npm run cli -- data-patches:push`.

Testing:

| Command | Description |
|---|---|
| `npm run test` | Runs the default test command, currently `npm run test:unit`. |
| `npm run test:unit` | Runs Jest with `test/jest-unit.json`. |
| `npm run test:watch` | Runs Jest in watch mode. |
| `npm run test:coverage` | Runs Jest and writes coverage output. |
| `npm run test:debug` | Runs Jest in-band with the Node inspector enabled. |

### CLI Admin User Creation

You can create a user with the `admin` or `superAdmin` role using the following command:

```bash
npm run cli -- admin:create --admin-email=admin@example.com --admin-password=SecurePassword123 --admin-firstname=Admin --admin-lastname=Name --role=superAdmin
```

## Tests

- The active automated suite is unit-test only.
- Unit tests live under `test/unit`.
- `npm run test` currently delegates to `npm run test:unit`.

## Data Patches

For controlling data changes (such as seeding reference books, backfilling values, or setting default configurations) independently of Drizzle schema migrations, the project includes a lightweight data patch runner:

- All patch files are located in `api/data-patches/` and must follow the naming convention: `YYYYMMDDHHMM-short-kebab-description.ts`.
- Patches must export a `patch` object conforming to the `DataPatch` interface.
- Patches must be idempotent and safe to run multiple times.

To create a new patch template:
```bash
npm run cli -- data-patches:create --name=patch-name
```

If no name is provided, the command generates a random drizzle-style slug:
```bash
npm run cli -- data-patches:create
```

To list patches:
```bash
npm run cli -- data-patches:list
```

To run patches locally:
```bash
npm run cli -- data-patches:push
```

## Production Deployment Strategy


To run database migrations and data patches securely during deployment, follow these rules:

1. **Schema Migrations (`db:migration:run`)**:
   - Must be executed inside the **release job / build container** (where dev dependencies are present) or during the CI/CD deployment pipeline.
   - Do **NOT** run schema migrations inside the production API runtime container because `drizzle-kit` is a development dependency and is omitted in the production image.

2. **Data Patches (`data-patches:push`)**:
   - Must be executed as a subsequent CD deployment pipeline step.
   - Run patches through the centralized CLI: `npm run cli -- data-patches:push`.
   - Requires `npm run build:cli` to have generated `dist/scripts/cli.js`, `dist/scripts/cli/operations/run-data-patches.js`, `dist/data-patches/*.js`, and `dist/data-patches-manifest.json`.
   - `NODE_ENV=production` selects the compiled CLI path automatically.

Recommended CD deployment pipeline execution order:
```bash
npm run build
npm run build:cli
npm run cli -- db:migration:run
npm run cli -- data-patches:push
```

The repository does not currently include a GitHub Actions workflow. The API Dockerfile build stage is the deployment-facing build reference and runs `npm run build` followed by `npm run build:cli`; migration and data-patch commands are intended for a release job or CI/CD step, not the production runtime container.

## Notes

- Product requirements and feature intent live in `../MVP_SCOPE.md`.
- `README_UA.md` is the Ukrainian companion document and should be kept aligned when behavior changes.
