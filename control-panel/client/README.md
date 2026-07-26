# Gym Marketplace Client

Next.js frontend for the sports coaching marketplace MVP.

## Current Scope

- App Router structure with dedicated public, athlete, coach, and admin areas
- Cookie-based authentication flows for athlete, coach, and admin entry points
- Frontend API proxying through `src/app/api/[...slug]/route.ts`
- Route/session access handling through `src/proxy.ts` and auth session services
- Coach and athlete chat workspaces with Socket.IO realtime connectivity
- Profile onboarding and account/profile management flows
- Program creation, editing, sharing, and progress views for coaches and athletes
- Trainer discovery and trainer profile views
- Reviews, feedback, sport data, theme handling, and shared UI primitives
- React Query data layer and Vitest test coverage for critical frontend modules

## Architecture Notes

- `src/app` contains route segments for `(public)`, `(auth)`, `athlete`, `coach`, and `admin`
- `src/features` contains domain modules such as `module-auth`, `module-chat`, `module-profile`, `module-program`, and `module-trainer`
- `src/lib/providers/providers.tsx` initializes shared client providers, including React Query
- `next.config.ts` defines security headers and rewrites for `/chat-realtime`

## Commands

```bash
npm install
npm run dev
npm run build
npm run start
npm run lint
npm run test
npm run test:watch
```

## Notes

- The client expects `API_BASE_URL`, `NEXT_PUBLIC_API_URL`, and `NEXT_PUBLIC_API_PORT` in `client/.env`.
- Product requirements and feature intent live in `../MVP_SCOPE.md`.
