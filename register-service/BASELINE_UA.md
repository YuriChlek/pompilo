# Register Service Baseline

Зафіксовано перед завершальним hardening фаз 0–10.

## Toolchain

- Node.js: `24.14.0`
- npm: `11.9.0`
- API: NestJS 11, TypeScript, Drizzle, PostgreSQL, Redis, BullMQ
- Client: Next.js 16, React 19, Vitest

## Відтворювані перевірки

```bash
make build-api
make build-client
make test-api
make test-client
make docker-config-dev
make docker-config-prod
```

Окремі API-команди:

```bash
cd register-service/api
npm run build
npm run test:unit -- --runInBand
```

Окремі client-команди:

```bash
cd register-service/client
npm run build
npm run test
```

## Зафіксовані результати

- API build: успішний.
- API unit suite: 176 suites, 1351 tests перед фінальним hardening; 1354 після нього.
- Client build: успішний, 40 routes.
- Client suite: 58 files, 297 tests.
- Root compose config: синтаксично валідний; env names і профіль `identity` узгоджені у фазі 3.

Числа тестів можуть зростати в наступних фазах; регресією вважається падіння раніше зеленого тесту або зникнення критичного coverage без затвердженої заміни.

## Runtime identity endpoints до hardening

- customer auth: register, legacy athlete/coach register, login, checkpoint verify/resend, logout, refresh, me, verify/resend email;
- admin auth: login, checkpoint verify/resend, logout, refresh, me;
- account security: password recovery/change, email change, sessions, reauth, deactivate/delete;
- mail administration: settings, setup state, template preview, test email;
- internal: health і metrics.

Повний versioned цільовий контракт знаходиться в `IDENTITY_CONTRACTS_V1_UA.md`.

## Background jobs

Identity/mail jobs, які потрібно зберегти:

- token/session/known-device/security-event cleanup — daily;
- password/email challenge cleanup — hourly;
- user deletion cleanup — daily;
- mail outbox relay — PostgreSQL LISTEN/NOTIFY + 5-minute fallback;
- stale mail lock recovery — every minute;
- mail outbox retention cleanup — daily at 03:00;
- BullMQ mail processor.

Marketplace jobs, які ще присутні до пізніших cleanup-фаз:

- media upload cleanup;
- chat presence processor та realtime timers.

## Відомі обмеження baseline

- gym marketplace modules ще підключені;
- legacy roles і endpoints ще доступні через compatibility layer;
- старий migration chain містить gym/media tables;
- спільна infrastructure topology перенесена в root `infra/`;
- clean identity migration baseline запланований у фазі 18.
