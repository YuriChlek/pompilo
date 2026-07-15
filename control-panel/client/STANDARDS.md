# Client Standards

## Purpose

This document defines the required naming and structural conventions for the `client` application.

The goal is consistency:

- predictable file locations
- predictable file names
- predictable ownership of types
- minimal ambiguity during refactoring

## Naming Rules

### General

- Use `kebab-case` for all `.ts`, `.tsx`, `.css` file names.
- Use `PascalCase` for React components and exported classes.
- Use `camelCase` for functions, variables, hooks internals, and constants that are not global-style constants.
- Use `SCREAMING_SNAKE_CASE` only for true constants that behave like static configuration values.

### Required file suffixes

- Interfaces: `*.interfaces.ts`
- Types: `*.types.ts`
- Enums: `*.enums.ts`
- Config: `*.config.ts`
- Server helpers: descriptive kebab-case, for example `get-current-user.ts`
- Utility files: descriptive kebab-case, for example `route-access.ts`

### Reserved framework names

Do not rename framework-reserved files:

- `page.tsx`
- `layout.tsx`
- `route.ts`
- `loading.tsx`
- `error.tsx`
- `not-found.tsx`
- `index.ts`
- `proxy.ts`

## Feature Module Structure

Each feature under `src/features` should use this structure when applicable:

```text
module-foo/
├── api-service/
├── components/
├── config/
├── enums/
├── hooks/
├── interfaces/
├── lib/
├── server/
└── types/
```

Not every module must contain every folder, but new code must follow this shape when the concern exists.

## What Goes Where

### `interfaces/`

Use `interfaces/` for stable object contracts:

- API entities
- service contracts
- DTO-like frontend payloads
- shared component prop contracts when they are reused across files

Examples:

- `User`
- `TradingAccount`
- `ApiKeyPayload`
- `TradingAccountService`

### `types/`

Use `types/` for:

- unions
- mapped types
- helper types
- local shared props types
- result/context/scope helper types

Examples:

- `TradingAccountAnalyticsPeriod`
- `AuthScope`
- `RouteContext`
- `RefreshResult`

### `enums/`

Use `enums/` only for enums.

Examples:

- `UserRoles`
- `COOKIE_NAMES`
- `MenuTypes`

### `config/`

Use `config/` for static module configuration objects.

Examples:

- auth scope map
- menu config
- route access config

### `lib/`

Use `lib/` for pure module utilities:

- formatters
- small deterministic helpers
- path or route helpers

Do not place domain contracts in `lib/`.

### `api-service/`

Use `api-service/` only for transport-level communication with backend endpoints.

Rules:

- no UI logic
- no React hooks
- no component state
- normalize server responses here when possible

Required structure:

```text
api-service/
├── client/
│   └── index.ts
└── server/
    └── index.ts
```

`api-service/client` is the browser/client transport boundary. It must use `HttpClient`
through `apiClient` from `@/lib/http-client/http-client`. Use it from Client Components,
React Query hooks, and event-driven browser flows. Upload flows that require `FormData`
may use the client refresh helper from `@/lib/http-client/fetch-with-auth-refresh`.
Do not import client transport from the root `@/lib/http-client` barrel.

`api-service/server` is the React Server Component transport boundary. It must use
`ServerHttpClient` through `getApiServerClient()` from
`@/lib/http-client/http-client.server`. Use it from pages, layouts, and server-only data
composition. Do not call `fetch` directly from feature server loaders when the request is
a normal JSON API request.
Do not import server transport from the root `@/lib/http-client` barrel.

Shared endpoint path decisions, response unwrapping, fallback behavior, and role-to-path
mapping belong in `api-service/client` or `api-service/server`, not in pages, components,
or hooks.

The feature-level `server/` folder is reserved for Next proxy/session infrastructure only.
Do not place domain data loaders such as `get-programs.ts`, `get-current-profile.ts`, or
`get-trainers.ts` in `features/*/server`. Put those functions in `api-service/server`.

### Auth and Session

Authentication is split across two scopes:

- `customer` for athlete and coach users.
- `admin` for admin and super-admin users.

Scope-specific cookie names, login paths, dashboard paths, and refresh paths must live in
`features/module-auth/config/auth-scope.config.ts`. Do not duplicate scope-to-cookie or
scope-to-path mapping in pages, components, hooks, or feature services.

The Next proxy is the first auth/session boundary:

- `src/proxy.ts` must stay thin and delegate auth handling to
  `features/module-auth/server/handle-auth-proxy.ts`.
- Proxy code must ignore static assets and API proxy routes.
- `handleAuthProxy` owns route-level auth redirects, pre-emptive refresh, and forwarding
  refreshed cookies to both the browser response and downstream Server Components.
- If refresh succeeds without a redirect, the proxy must merge returned `Set-Cookie`
  values into the downstream request `cookie` header so Server Components can see the
  refreshed session during the same render.

Session refresh rules:

- Refresh is attempted when the access token is missing or expired according to the JWT
  payload check.
- The JWT payload check is only a pre-emptive client-side routing optimization. The backend
  remains the final authority for token signature, validity, role, and permissions.
- Refresh requests must send only the refresh cookie for the relevant auth scope.
- Normal API requests must not forward refresh cookies.

API proxy cookie rules:

- `app/api/[...slug]/route.ts` must use `filterApiProxyCookieHeader` from
  `features/module-auth/server/auth-cookie-header.ts` before forwarding requests to the
  backend.
- Normal customer API requests may forward only the customer access cookie and allowed
  pass-through cookies such as `theme`.
- Normal admin API requests may forward only the admin access cookie and allowed
  pass-through cookies such as `theme`.
- Customer refresh requests may forward only the customer refresh cookie.
- Admin refresh requests may forward only the admin refresh cookie.

Server session rules:

- Server Components must not infer an authenticated customer only from cookie presence.
- Use `getActiveCustomerSession()` for request-scoped customer session resolution.
- `getActiveCustomerSession()` may use cookie presence as a fast gate, but the active
  customer and role must be resolved through the backend `/me` response.
- Admin session state should remain separate from customer session state.
- Do not mix admin and customer cookies in a single backend request.

HTTP client auth rules:

- Server-side API calls must use `getApiServerClient()` from
  `@/lib/http-client/http-client.server`.
- Server-side requests must forward only the access cookie for the resolved or explicitly
  configured auth scope.
- When a server-side request receives `401`, the server HTTP client may refresh the matching
  scope and retry with the access cookie returned by the refresh response.
- Browser JSON API calls must use `apiClient` from `@/lib/http-client/http-client`.
- Browser upload or `FormData` flows may use `fetchWithAuthRefresh`.
- New auth-aware API helpers should accept or derive an `authScope` instead of guessing from
  arbitrary UI state.

### Realtime presence

Online/offline presence is a user-level realtime state, not a chat-page state.

Rules:

- A user is `online` only while they have at least one active role-scoped websocket connection.
- A valid login cookie or refreshable session does not mean the user is online.
- A user becomes `offline` when their last role-scoped websocket disconnects or the backend presence TTL expires.
- Opening a specific chat conversation must not be required for online/offline presence to work.
- Conversation-level websocket events such as join, read receipts, typing, and message updates belong to chat UI behavior, not to the definition of global presence.
- Athlete and coach role areas should own their own role-scoped realtime connection.
- Chat UI should consume shared presence state by participant user ID and display `Status unavailable` only when the realtime connection itself is unavailable.
- `ChatRealtimeProvider` owns socket creation, connection, disconnect, and global presence listeners.
- Conversation hooks such as `useChatConversationRealtime` may attach conversation listeners and emit join/read/typing events, but must not call `io()` or disconnect the role-level socket.

### `hooks/`

Use `hooks/` for React Query hooks and stateful UI hooks.

Rules:

- hooks may compose services
- hooks should not contain large view logic
- query normalization should happen before JSX when possible

### `components/`

Use `components/` for React UI only.

Rules:

- one exported component per file
- if a helper component grows beyond trivial size, move it to its own file
- keep files focused on rendering and interaction

## Component Rules

- Prefer one exported component per `.tsx` file.
- Private helper components may stay in the same file only if they are very small and not reused.
- Shared component prop types must live in `types/` or `interfaces/`, not randomly inside multiple components.
- Avoid mixing data shaping and rendering in the same large component.

## Import Rules

- Import enums from `enums/`.
- Import interfaces from `interfaces/`.
- Import helper types from `types/`.
- Avoid importing from old file names after rename.
- Prefer direct imports over broad barrel usage unless the barrel is a deliberate public API.

## Responsive Rules

- Write component CSS mobile-first.
- Base styles must represent the smallest supported layout of the component.
- Expand layouts progressively with `min-width` media queries only.
- Use only the canonical breakpoint scale:
  - `768` for tablet+
  - `1024` for laptop+
  - `1280` for desktop+
  - `1536` for wide+
- Do not introduce ad hoc breakpoint values in component CSS.
- Do not use `max-width` breakpoints in component styles.
- When migrating an existing component from `max-width` to `min-width`, first move the narrow layout into the base styles, then restore wider layouts through `min-width`.
- Do not combine breakpoint migration with visual redesign unless explicitly requested.
- Run `npm run lint:responsive` after responsive CSS changes or before opening a PR that touches component styles.

## Build Tooling

- This client uses Next.js with Turbopack.
- Add bundler-specific behavior only through `turbopack` configuration in `next.config.ts`.
- Do not add `webpack` configuration, webpack loaders, or webpack-only plugins to this project.
- If a package documents only webpack setup, translate it to a Turbopack rule or choose a Turbopack-compatible alternative.

## Data Handling Rules

- Do not spread `response.data?.…` through components.
- Normalize nullable or optional response data at the service or hook boundary.
- Prefer typed fallback objects over repeated optional chaining in JSX.

## Tests

- Test fixtures should use the same domain interfaces/types as production code.
- Test-only probe components may live inside test files.
- Do not let test helpers define conventions for production structure.

## Refactor Policy

When touching a module:

1. Keep naming aligned with this document.
2. Move newly introduced shared contracts into the correct folder.
3. Update imports immediately after rename or extraction.
4. Run client tests after structural changes.

## Review Checklist

Before merging frontend code, verify:

- file names are `kebab-case`
- enums are only in `enums/`
- reusable interfaces are in `interfaces/`
- helper/shared types are in `types/`
- no oversized multi-component files without reason
- no stale imports to renamed files
- no repeated `data?.…` usage where typed normalization is possible
