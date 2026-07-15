# Gym Marketplace API

NestJS бекенд для MVP маркетплейсу спортивного коучингу.

## Поточний Обсяг

- Cookie-based автентифікація для `athlete`, `coach`, `admin` і `superAdmin`
- Окремі role entry points через `module-customer-auth` і `module-admin-auth`
- Спільна інфраструктура для user, password, token і refresh-token
- Frontend-facing модулі для зон athlete, coach і admin
- Чат-розмови, доставка повідомлень, typing, presence і realtime-оновлення через Socket.IO
- Профілі coach і athlete, завантаження фото профілю та роздача статики через `/uploads`
- Шаблони програм, флоу поширення/копіювання, завершення днів програми та feedback по вправах
- Пошук тренерів, каталог видів спорту, reviews, feedback, media assets, moderation і translation services
- Збереження в PostgreSQL через Drizzle ORM та інтеграція з Redis/BullMQ
- Swagger увімкнений на `/swagger`
- Покриття core services і realtime-інфраструктури unit-тестами

## Нотатки По Архітектурі

- `src/app.module.ts` підключає доменні модулі та глобальний response interceptor
- `src/main.ts` налаштовує validation, cookies, CORS, Swagger, роздачу статики uploads і Redis-backed Socket.IO adapter
- Drizzle ORM використовує SQL-міграції через drizzle-kit, а схеми описуються в кожному бізнес-модулі окремо
- Redis використовується і для BullMQ, і для міжінстансового Socket.IO messaging, з локальним fallback, якщо Redis недоступний

## Відповідальність Auth Модулів

Точки входу автентифікації розділені за сімействами ролей:

- `module-customer-auth` відповідає за неадмінські ролі: `athlete` і `coach`.
- `module-admin-auth` відповідає за адмінські ролі: `admin` і `superAdmin`.

Для неадмінських ролей логін, реєстрація, logout, refresh token і current-user lookup обслуговуються через `module-customer-auth`.
Для адмінських ролей логін, logout, refresh token і current-user lookup обслуговуються через `module-admin-auth`.

Не додавайте маршрути автентифікації athlete або coach до `module-admin-auth`, і не додавайте адмінські auth-маршрути до `module-customer-auth`.

Admin і customer authentication cookies використовують `path: '/'`, щоб браузер надсилав їх
до Next.js `/api/*` proxy, через який frontend звертається до backend. Для auth scopes
використовуються різні імена cookies.

Cookie `Path` не є межею безпеки між admin і customer scopes. Звичайні API-запити повинні
нести лише access cookie цільового scope, refresh endpoints — лише відповідний refresh cookie,
а backend повинен перевіряти token realm і role. Frontend proxy та server-side fetch helpers
мають фільтрувати cookie headers, а не пересилати всі cookies без розбору.

## Environment Variables

Під час NestJS bootstrap environment variables проходять централізовану перевірку.
Відсутня або некоректна core-конфігурація зупиняє запуск із повним списком помилок.

Обов’язкові core variables:

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

Для `NODE_ENV`, HTTP, Redis і mail capacity settings використовуються задокументовані
development defaults. Числові та boolean-значення нормалізуються до передачі через
`ConfigService`.

## Команди

Усі команди нижче виконуються з папки `api/`.

Встановлення залежностей:

```bash
npm install
```

Життєвий цикл застосунку:

| Команда | Опис |
|---|---|
| `npm run build` | Збирає NestJS застосунок у `dist/`. |
| `npm run start` | Одноразово запускає NestJS застосунок через Nest CLI. |
| `npm run start:dev` | Запускає API в development-режимі з watch mode. |
| `npm run start:debug` | Запускає API з Node debugger і watch mode. |
| `npm run start:prod` | Запускає зібраний production entrypoint із `dist/src/main` та `NODE_ENV=production`. Перед цим потрібно виконати `npm run build`. |

Якість коду:

| Команда | Опис |
|---|---|
| `npm run format` | Форматує `src/**/*.{ts,tsx}` і `test/**/*.{ts,tsx}` через Prettier. |
| `npm run lint` | Запускає ESLint для source/test/script файлів із `--fix`. |

База даних і seed-скрипти:

| Команда | Опис |
|---|---|
| `npm run cli -- db:migration:generate [--name=<migration-name>]` | Генерує Drizzle migration із поточних schema-файлів. Без `--name` drizzle-kit використовує дефолтну згенеровану назву migration. Для повністю нового baseline generator також додає raw SQL для `mail_outbox` notification trigger/function і autovacuum settings. |
| `npm run cli -- db:migration:run` | Застосовує pending Drizzle migrations із поточного environment. |
| `npm run cli -- db:dev:migration:run` | Застосовує pending Drizzle migrations із `NODE_ENV=development`. |
| `npm run cli -- db:studio` | Відкриває Drizzle Studio для перегляду БД. |
| `npm run cli -- db:destructive-reset` | Використовує `DB_NAME` з environment, видаляє і створює заново схеми `public` і `drizzle`, запускає migrations, seed-ить default sports і опційно створює admin user, якщо задані всі admin seed env vars. Відмовляється працювати в production і для protected databases. |
| `npm run cli -- admin:create --admin-email=<email> --admin-password=<password> --admin-firstname=<firstname> --admin-lastname=<lastname> [--role=<role>]` | Створює користувача з роллю `admin` або `superAdmin`. За замовчуванням роль `admin`. |

## Operational CLI

Канонічний формат operational commands:

```bash
npm run cli -- <command> [--key=value]
```

У цій команді `--` після `npm run cli` є npm argument separator. Це не аргумент CLI command. Аргументи конкретних команд документуються у форматі `--key=value`. Єдиний framework-level виняток — довідка по команді у форматі `help <command>`.

Канонічні команди:

| Команда | Опис |
|---|---|
| `npm run cli -- help` | Показує список доступних CLI commands. |
| `npm run cli -- help <command>` | Показує usage для конкретної команди. |
| `npm run cli -- data-patches:push` | Запускає pending data patches. |
| `npm run cli -- data-patches:list` | Показує data patches та їхній статус. |
| `npm run cli -- data-patches:dry-run` | Показує pending data patches без застосування. |
| `npm run cli -- data-patches:create --name=patch-name` | Створює template data patch. |
| `npm run cli -- data-patches:build` | Компілює data patches і генерує manifest. |
| `npm run cli -- db:migration:generate` | Генерує Drizzle migration з дефолтною згенерованою назвою. |
| `npm run cli -- db:migration:generate --name=add-user-avatar` | Генерує Drizzle migration. |
| `npm run cli -- db:migration:run` | Застосовує pending Drizzle migrations. |
| `npm run cli -- db:dev:migration:run` | Застосовує pending Drizzle migrations із `NODE_ENV=development`. |
| `npm run cli -- db:studio` | Запускає Drizzle Studio. |
| `npm run cli -- db:destructive-reset` | Скидає налаштовану development БД, запускає migrations і seed data. |
| `npm run cli -- admin:create --admin-email=admin@example.com --admin-password=Seed1234 --admin-firstname=Admin --admin-lastname=Name --role=superAdmin` | Створює admin user. |
| `npm run cli -- demo-data:push` | Seed-ить demo data для development. |

Operational npm wrappers видалені. Нові operational commands треба додавати як CLI command classes, реєструвати в CLI registry і документувати через entrypoint `npm run cli -- <command>`.

Deployment steps мають використовувати `npm run build`, `npm run build:cli`, `npm run cli -- db:migration:run` і `npm run cli -- data-patches:push`.

Тести:

| Команда | Опис |
|---|---|
| `npm run test` | Запускає default test command, зараз це `npm run test:unit`. |
| `npm run test:unit` | Запускає Jest із конфігом `test/jest-unit.json`. |
| `npm run test:watch` | Запускає Jest у watch mode. |
| `npm run test:coverage` | Запускає Jest і генерує coverage output. |
| `npm run test:debug` | Запускає Jest in-band із Node inspector. |

`db:migration:generate` використовує snapshot для кожного journal entry. Під час генерації
повністю нового baseline project generator також додає PostgreSQL trigger/function і налаштування
autovacuum для `mail_outbox`, які не можуть бути описані Drizzle schema snapshot.

## Email Delivery Quota

- `DB_POOL_MAX` задає PostgreSQL pool на одну API replica, default і верхня межа — `30`.
- Redis у Docker stack має initial limit `2GB` та eviction policy `noeviction`.
- Повні capacity targets, правила узгодження provider quota та load/stress acceptance criteria
  описані у [`../docs/email_capacity_limits.md`](../docs/email_capacity_limits.md).
- `MAIL_QUEUE_LIMITER_MAX` і `MAIL_QUEUE_LIMITER_DURATION` задають BullMQ throughput window.
- Поточний default для local/development: `100` jobs за `1000ms`, тобто provisional target до 100 critical emails/sec або 6,000 emails/minute.
- Для production ці значення мають бути не вищими за реальну SMTP provider quota конкретного account.
- `MAIL_WORKER_CONCURRENCY` задає паралельність worker-а на одну API replica; default `4`.
- `MAIL_RETRY_ATTEMPTS`, `MAIL_RETRY_BACKOFF_TYPE`, `MAIL_RETRY_BACKOFF_DELAY` і `MAIL_RETRY_BACKOFF_JITTER` задають retry policy. Кількість спроб за замовчуванням `3`; jitter є ratio `0..1`, default `0.2`.

### Створення адміністратора через CLI

Ви можете створити користувача з роллю `admin` або `superAdmin` за допомогою команди:

```bash
npm run cli -- admin:create --admin-email=admin@example.com --admin-password=Seed1234 --admin-firstname=Admin --admin-lastname=Name --role=superAdmin
```

## Патчі даних (Data Patches)

Для здійснення контрольованих змін даних (таких як посів довідників, оновлення/заповнення значень, встановлення дефолтних конфігурацій) незалежно від Drizzle-міграцій схем, у проекті передбачено легковисний раннер патчів:

- Файли патчів знаходяться в `api/data-patches/` та повинні називатися за шаблоном: `YYYYMMDDHHMM-short-kebab-description.ts`.
- Патч повинен експортувати об'єкт `patch`, який відповідає інтерфейсу `DataPatch`.
- Патчі повинні бути ідемпотентними та безпечними для повторних запусків.

Створення нового шаблону патчу:
```bash
npm run cli -- data-patches:create --name=patch-name
```

Якщо назву не передано, команда згенерує випадковий slug у стилі Drizzle:
```bash
npm run cli -- data-patches:create
```

Перегляд списку патчів:
```bash
npm run cli -- data-patches:list
```

Локальний запуск патчів:
```bash
npm run cli -- data-patches:push
```

## Стратегія розгортання в продакшені


Для безпечного запуску міграцій бази даних та патчів під час деплою дотримуйтесь таких правил:

1. **Міграції схеми (`db:migration:run`)**:
   - Повинні запускатися у **release job / build container** (там, де доступні dev dependencies) або під час кроків CI/CD збірки.
   - **НЕ** запускайте міграції всередині продуктового API runtime контейнера, оскільки `drizzle-kit` є dev dependency і відсутній у фінальному image.

2. **Патчі даних (`data-patches:push`)**:
   - Повинні запускатися як наступний крок CD пайплайну розгортання.
   - Запуск патчів виконується через centralized CLI: `npm run cli -- data-patches:push`.
   - Перед цим `npm run build:cli` має згенерувати `dist/scripts/cli.js`, `dist/scripts/cli/operations/run-data-patches.js`, `dist/data-patches/*.js` і `dist/data-patches-manifest.json`.
   - `NODE_ENV=production` автоматично обирає compiled CLI path.

Рекомендований порядок кроків деплою:
```bash
npm run build
npm run build:cli
npm run cli -- db:migration:run
npm run cli -- data-patches:push
```

У репозиторії зараз немає GitHub Actions workflow. API Dockerfile build stage є deployment-facing build reference і запускає `npm run build`, потім `npm run build:cli`; migrations і data patches мають запускатися в release job або CI/CD step, а не в production runtime container.

## Примітки

- API очікує змінні для database, JWT, encryption, cookies і Redis у `api/.env`.
- Вимоги до продукту та задум функціональності описані в `../MVP_SCOPE.md`.
