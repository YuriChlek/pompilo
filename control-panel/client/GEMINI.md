# Project Instructions for Gemini CLI

## Core Mandate: Standards Compliance
Before any code modification or architectural proposal, YOU MUST READ AND ADHERE TO `STANDARTS.md`.

## Critical Architectural Rules to Remember:
1.  **Feature Structure:** Adhere to the prescribed folder structure within features (`api-service/`, `components/`, `hooks/`, `interfaces/`, etc.).
2.  **API Transport Separation:** Use `api-service/client` for Client Components/Hooks and `api-service/server` for Server Components. NEVER mix them.
3.  **Responsive Design:** Use ONLY mobile-first `min-width` media queries with canonical breakpoints (768, 1024, 1280, 1536). `max-width` is strictly forbidden.
4.  **Real-time Presence:** Presence status is user-level and driven by active WebSocket connections, not by the mere existence of cookies.
5.  **Naming Conventions:** Use `kebab-case` for all files (`*.tsx`, `*.ts`, `*.css`). Use required suffixes for contracts (`*.interfaces.ts`, `*.types.ts`, `*.enums.ts`).
6.  **Data Normalization:** Normalize server responses at the service or hook boundary. Avoid spreading raw response data through components.
7.  **Bundler Compliance:** This project uses Next.js with Turbopack. Do not add Webpack-specific configurations or loaders.

## Workflow:
- Always verify the feature structure against `STANDARTS.md` before adding new modules or components.
- Ensure all API calls use the appropriate transport boundary (client vs server).
- Run `npm run lint:responsive` after any CSS changes to ensure compliance with mobile-first rules.
