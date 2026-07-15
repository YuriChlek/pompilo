# API Standards

## Purpose

This document defines the required naming and structural conventions for the NestJS `api` application.

The goal is consistency:

- modules look the same
- contracts live in predictable places
- integration code is separated from domain code
- renames and refactors remain low-risk

## Naming Rules

### General

- Use `kebab-case` for folders.
- Use `kebab-case` for file names unless Nest requires a suffix-oriented class file pattern.
- Use `PascalCase` for classes, DTOs, schema exports, services, controllers, guards, and modules.
- Use `camelCase` for functions and variables.

### Required file suffixes

- Module: `*.module.ts`
- Service: `*.service.ts`
- Controller: `*.controller.ts`
- Guard: `*.guard.ts`
- Strategy: `*.strategy.ts`
- DTO: `*.dto.ts`
- Drizzle schema: `*.schema.ts`
- Interface contract: `*.interfaces.ts`
- Enum: `*.enums.ts`
- Config factory: `*.config.ts`
- Pure util: descriptive kebab-case, for example `setup-swagger.util.ts`

## Module Structure

Each backend domain module should follow this shape when applicable:

```text
module-foo/
├── controllers/
├── dto/
├── schemas/
├── enums/
├── interfaces/
├── repository/
├── services/
├── guards/
├── strategies/
├── constants/
├── types/
├── events/
└── module-foo.module.ts
```

Not every module needs every folder, but new code should follow this shape when the concern exists.

## What Goes Where

### `controllers/`

Use `controllers/` for NestJS controllers.

Rules:

- receive request data
- delegate to services
- not contain business rules
- not contain query-building logic

### `dto/`

Use `dto/` only for transport-layer request and response DTOs.

Rules:

- validate incoming data here
- do not place business logic here

### `schemas/`

Use `schemas/` for Drizzle ORM schema files that belong to a specific business module.

Rules:

- business schemas must live in their owning module, for example `src/module-user/schemas/users.schema.ts`
- do not place business Drizzle schemas inside `module-drizzle`
- each module should own and version its own tables, enums, and schema-related exports
- use `*.schema.ts` for Drizzle schema files
- if a module has Drizzle schemas, add a local `schemas/index.ts` export for that module
- do not manually edit generated migration files during feature work; update Drizzle schema files first, then regenerate migrations from the schema
- if a generated migration contains unrelated changes, fix the migration metadata/snapshot baseline and regenerate instead of hand-editing the migration SQL

### `interfaces/`

Use `interfaces/` for stable contracts:

- integration response shapes
- auth/token payload contracts
- service-facing structural contracts

### `types/`

Use `types/` for helper types, raw query result shapes, mapped types, and technical type aliases.

### `enums/`

Use `enums/` only for enums.

Rules:

- use enums for closed domain values and state machines
- do not use enums for seed lists, default catalog entries, or module bootstrap data

### `events/`

Use `events/` for module-specific events, event handlers, and event listeners.

### `services/`

Use `services/` for business logic and infrastructure coordination.

Rules:

- one primary responsibility per service
- regular services must not work with the database directly
- regular services must not inject Drizzle DB clients or inline SQL/query builders directly
- database access belongs in `repository/` classes only
- services orchestrate repository classes instead of persistence primitives
- external API integration should not be mixed with controller logic

### `repository/`

Use `repository/` for persistence-only classes that work with the database.

Rules:

- all direct database access belongs here
- Drizzle queries and persistence-specific SQL belong here
- repository classes may inject the shared Drizzle DB provider and compose database queries
- repository classes must not contain orchestration, auth logic, or unrelated integration logic
- regular services should depend on repository classes, not on Drizzle primitives directly

### `constants/`

Use `constants/` for stable domain constants and module-level config values.

Rules:

- use `constants/` for seed data, default lists, and bootstrap-ready readonly config
- if the value is consumed as structured data rather than compared as a finite state, prefer `constants/` over `enums/`
- for catalog data that must be materialized into SQL migrations, keep the source list in TypeScript constants and generate the final seed SQL through project migration tooling rather than hand-editing `.sql` files

### `config/`

Use `config/` for Nest config factories and infra setup.

Examples:

- Drizzle config
- JWT config
- BullMQ config

### `module-drizzle`

`module-drizzle` is infrastructure only.

Rules:

- `module-drizzle` provides the shared Drizzle DB provider and DB config
- `module-drizzle` may aggregate schema exports from business modules
- `module-drizzle` must not become the owner of business tables or business schema definitions
- drizzle-kit config should collect schema files across project modules rather than pointing only to `module-drizzle`

### `common/`

Use `common/` only for cross-module shared infrastructure:

- interceptors
- shared DTO wrappers
- generic utilities

Do not put domain-specific logic into `common/`.

## Architectural Boundaries

### Authentication module ownership

Auth entry points are split by role family and must stay that way:

- `module-customer-auth` handles non-admin roles: `athlete` and `coach`.
- `module-admin-auth` handles admin roles: `admin` and `superAdmin`.

The following flows for non-admin roles belong to `module-customer-auth`:

- login
- registration
- logout
- token refresh
- current-user lookup

The following flows for admin roles belong to `module-admin-auth`:

- login
- logout
- token refresh
- current-user lookup

Do not introduce athlete or coach auth endpoints into `module-admin-auth`.
Do not introduce admin or super-admin auth endpoints into `module-customer-auth`.

### Authentication cookie paths

Authentication cookies use `path: '/'` because browser requests reach the backend through
the Next.js `/api/*` proxy. A cookie scoped to `/admin` would not be sent to frontend proxy
routes such as `/api/admin/*`.

Rules:

- customer and admin access cookies use `path: '/'`
- customer and admin refresh cookies use `path: '/'`
- admin and customer credentials must use distinct cookie names
- cookie `Path` is a delivery mechanism, not the security boundary between auth scopes
- scope isolation must be enforced by frontend cookie filtering, backend token realm validation,
  role authorization, and refresh endpoint ownership
- do not narrow admin cookies to `/admin` unless frontend API proxy routes and session handling
  are redesigned at the same time

### Authentication cookie forwarding

Refresh tokens are restricted credentials and must not be forwarded with normal API requests.

Rules:

- normal authenticated API requests may forward access cookies only
- refresh API requests may forward only the refresh cookie required for that auth scope
- frontend proxy and server-side fetch code must filter cookie headers before forwarding requests to the API
- do not rely only on browser cookie `Path` matching to protect refresh cookies, because server-side code can manually construct cookie headers
- customer refresh cookies must be removed from non-refresh `/api/*` proxy requests
- admin refresh cookies must be removed from non-refresh `/api/admin/*` proxy requests
- normal customer requests must not forward admin cookies, and normal admin requests must not forward customer cookies
- refresh requests should include only the minimal headers needed for refresh-token validation, such as `cookie`, `user-agent`, `x-forwarded-for`, and `x-real-ip`
- direct server-side fetch helpers must use a shared cookie-header builder instead of manually forwarding `cookies().getAll()`

### Chat realtime presence

Chat presence is based on active websocket sockets, not on the mere existence of an access or refresh cookie.

Rules:

- A user is online when at least one authenticated websocket socket for that role is registered.
- A user is offline only after the last socket for that role is removed or its presence TTL expires.
- The source of truth for online/offline is the presence socket registry, currently backed by Redis.
- `conversation:join` must not be required to make a user online.
- On websocket connection, the gateway should emit a current presence snapshot for the authenticated user's chat counterparts so UI status does not depend on opening a specific conversation first.
- `conversation:join` may emit a current participant presence snapshot for the opened conversation, but it is not the presence lifecycle owner.
- Websocket authentication must be role-aware because the same browser can carry athlete and coach cookies at the same time.
- Presence broadcasts should be user-level events sent to relevant counterpart user rooms on first socket connect and last socket disconnect.

### Controllers

Controllers should:

- receive request data
- delegate to services
- not contain business rules
- not contain query-building logic

### Services

Services should:

- own business decisions
- orchestrate repositories, tokens, encryption, queues, and integrations
- throw meaningful HTTP exceptions only at clear boundaries

### Repository-style services

Do not keep repository wrappers inside regular `services/`.
If a class is effectively a repository wrapper, move it to `repository/` and keep it focused:

- persistence only
- no orchestration
- no auth logic
- no external API logic

### Integration services

External exchange clients should be isolated from domain orchestration as much as possible.

Examples:

- request signing
- pagination handling
- response mapping

These concerns should not leak across unrelated modules.

## Naming Consistency Rules

The same concept must have the same name across the project.

Examples of what must be cleaned and avoided:

- no mixed `analyse` vs `analyze`
- no technical typos like `tocken`
- no inconsistent queue names like misspelled `excange`

If a term is chosen, use it everywhere:

- folder names
- file names
- DTO names
- schema names
- service method names

## Import Rules

- Import enums from `enums/`.
- Import interfaces from `interfaces/`.
- Import helper types from `types/`.
- Avoid circular imports between modules.
- Prefer direct imports over broad barrels unless the barrel is intentional and stable.

## Error Handling Rules

- Do not swallow unexpected errors silently.
- Re-throw known HTTP errors.
- Wrap unknown failures with clear module-specific messages.
- Avoid duplicating generic try/catch blocks unless they add boundary meaning.

## Database Performance Rules

- **Foreign Key Indexing Rule**: Every new FK column must be reviewed for indexing. Most non-trivial FK columns should be indexed unless the table is guaranteed to be tiny or access patterns definitively show it is unnecessary.
- **Index Strategy Rule**: Every hot repository method utilizing `WHERE` + `ORDER BY` must be reviewed against a concrete composite or partial index strategy.

## Testing Rules

- Unit tests cover service logic and pure behaviors.
- Integration tests cover repository + module behavior.
- E2E tests cover key auth and API flows.
- Test file names must mirror the production unit under test.

## Refactor Policy

When touching a backend module:

1. Keep naming aligned with this document.
2. Move contracts into the correct folder.
3. Do not introduce new mixed-purpose files.
4. Rename technical typos when touching affected areas.
5. Run relevant tests after structural changes.

## Data Patches Rules

To maintain safe, idempotent, and auditable data changes, follow these rules:

### Database Schema vs. Data Patches
- **Drizzle schema migrations** are responsible for **structural (schema) changes**: tables, columns, indexes, constraints, and foreign keys.
- **Data patches** are responsible for **data changes**: seeding reference lists, backfilling old rows, system/default records setup, data normalization, and one-time production fixes. Data patches must never mutate the database schema.

### Naming Conventions
- Data patch files must be created in `api/data-patches/` and named: `YYYYMMDDHHMM-short-kebab-description.ts`.
- Prefer `npm run cli -- data-patches:create --name=patch-name` to create new patch templates. Without a provided name, the generator creates a random drizzle-style slug.
- The exported `patch.name` must exactly match the file name (without the `.ts` extension).
- Patch files must export a single `patch` object implementing the `DataPatch` interface.

### Safety and Transaction Rules
- Patches **must be idempotent**. They must be safe to execute multiple times (e.g., using `INSERT ... ON CONFLICT DO UPDATE`, or checking condition boundaries).
- Patches must work **within the provided transaction client** (`context.client`). Do **NOT** call `BEGIN`, `COMMIT`, or `ROLLBACK` manually inside patches.
- Patches must **not** import NestJS services, repositories, or components requiring dependency injection.
- Patches must **not** open custom database connections or rely on Redis, BullMQ, Socket.IO, or cron jobs.
- Once merged, **applied patch files must never be edited**. Any further updates must be introduced via a new patch file.

## Operational CLI Rules

- The canonical operational command form is `npm run cli -- <command> [--key=value]`.
- The `--` after `npm run cli` is the npm argument separator. It is not passed to the CLI command.
- Command-specific arguments must use `--key=value`. Do not document positional arguments or space-separated flag values for new commands.
- The only argument-format exception is framework help: `npm run cli -- help <command>`.
- New operational commands must be implemented as CLI command classes under `api/scripts/cli/commands/` and registered in `api/scripts/cli.ts`.
- Do not add operational npm wrapper scripts. The supported operational entrypoint is `npm run cli -- <command>`.
- `build:cli` is the source-only bootstrap build path and must not depend on `npm run cli -- data-patches:build`.
- Keep permanent package scripts limited to build, CLI bootstrap, start, test, lint, and format commands.

## Review Checklist

Before merging backend code, verify:

- file names follow suffix conventions
- controllers stay thin
- services are focused
- DTOs are only transport contracts
- Drizzle schemas live in `schemas/` inside their owning business module
- interfaces/types/enums live in their own folders
- naming is consistent across module boundaries
- no stale imports after refactor
- new data patches follow `YYYYMMDDHHMM-` naming and are fully idempotent
- no NestJS DI imports or manual transaction control inside data patches
