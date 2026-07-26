# Project Instructions for Gemini CLI

## Core Mandate: Standards Compliance
Before any code modification or architectural proposal, YOU MUST READ AND ADHERE TO `STANDARTS.md`.

## Critical Architectural Rules to Remember:
1.  **Repository Pattern:** NEVER inject Drizzle or write SQL in Services. Always use `repository/` classes for all direct database access.
2.  **Thin Controllers:** Controllers must only receive request data and delegate to services. They must not contain business rules or query-building logic.
3.  **Schema Ownership:** Drizzle schemas MUST live in their owning business module under `src/module-<name>/schemas/`. Do not place business schemas inside `module-drizzle`.
4.  **Auth Separation:** Keep `module-customer-auth` (athletes/coaches) and `module-admin-auth` (admins) strictly separated. Admin and customer cookies use `path: '/'` for the Next.js `/api/*` proxy, but must have distinct names and be isolated through proxy filtering, token realm validation, and role authorization.
5.  **Naming & Suffixes:** Follow naming conventions strictly (kebab-case for files, PascalCase for classes). Use required suffixes: `*.module.ts`, `*.service.ts`, `*.controller.ts`, `*.repository.ts`, `*.schema.ts`, `*.dto.ts`, etc.
6.  **Module Structure:** Adhere to the prescribed folder structure within modules (controllers, dto, schemas, services, repository, etc.).
7.  **Real-time Presence:** Chat online status is driven by active WebSocket sockets in Redis, not by session cookies.
8.  **Performance & Scalability:** 
    - Every new "hot" query path (WHERE/ORDER BY on large tables) MUST be reviewed with `EXPLAIN`.
    - Use cursor-based (keyset) pagination for all growing lists. Avoid `OFFSET`.
    - Keep write transactions small and scoped to a single entity/operation where possible.
    - Use `findByIdWithoutRelations` for access control checks to avoid heavy hydration.

## Workflow:
- Always verify the module structure against `STANDARTS.md` before adding new functionality.
- Ensure all transport-layer contracts are in `dto/` and domain contracts are in `interfaces/` or `enums/`.
- Maintain clear boundaries: Controllers -> Services -> Repositories.
