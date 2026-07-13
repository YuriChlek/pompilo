# План очищення `register-service`

## 1. Мета

Перетворити скопійований Pampilo Gym Marketplace на окремий Identity and Registration Service для торгової платформи.

Після очищення сервіс повинен відповідати лише за:

- реєстрацію користувачів;
- аутентифікацію: login, logout, access/refresh tokens, сесії;
- авторизацію: ролі, permissions і service-to-service access;
- підтвердження email;
- login checkpoints та повторне надсилання кодів;
- password reset і зміну пароля;
- зміну email;
- MFA або підготовлений extension point для MFA;
- керування активними сесіями та known devices;
- security events і аудит identity-операцій;
- надсилання системних email;
- мінімальний account lifecycle: deactivate, revoke sessions, delete/anonymize;
- видачу identity/entitlement даних trading service через підписані токени або події.

Сервіс не повинен містити спортивний marketplace, тренерів, атлетів, програми, чат, медіа, відгуки, модерацію чи object storage.

Цей документ описує план очищення. Він не є дозволом на автоматичне видалення файлів без виконання dependency-аудиту та тестів кожного етапу.

## 2. Поточний стан

`register-service` є майже повною копією Pampilo Gym Marketplace:

- NestJS API;
- Next.js client;
- PostgreSQL через Drizzle;
- Redis;
- BullMQ;
- SMTP/Mailpit і mail outbox;
- Nginx gateway;
- Loki, Alloy і Grafana;
- MinIO/GCS object storage;
- gym-домени: athlete, coach, trainer, sport, program, profile, review, chat, media, feedback, moderation.

Auth-код не ізольований від gym-домену:

- `UserRoles` містить `ATHLETE` і `COACH`;
- реєстрація має endpoints `register/athlete` і `register/coach`;
- DTO вимагають стать, дату народження, спеціалізацію та спортивні поля;
- `CustomerRegistrationRepository` разом із `users` створює athlete/coach profile і account setup state;
- password reset та email change exposed через athlete/coach account controllers;
- client registration flow містить athlete personalization і coach setup;
- загальний Drizzle schema index імпортує всі marketplace-схеми;
- старі міграції створюють таблиці media/object storage та інші непотрібні сутності.

Тому gym-модулі не можна безпечно видалити одним комітом. Спочатку потрібно створити нейтральні identity-контракти та перенести до них потрібні flows.

## 3. Цільова структура

```text
register-service/
  api/
    src/
      common/
        health/
        redis/
        rate-limiting/
        request-context/
      config/
      identity/
        auth/
        tokens/
        users/
        sessions/
        devices/
        security-events/
        account-recovery/
        authorization/
      mail/
        outbox/
        delivery/
        templates/
        settings/
      integration/
        trading-service/
        outbox-events/
      persistence/
        drizzle/
        schemas/
        migrations/
      logger/
  client/
    src/
      app/
        login/
        register/
        verify-email/
        forgot-password/
        reset-password/
        account/security/
      features/
        auth/
        account-security/
  # власної infra-папки після консолідації немає;
  # спільна інфраструктура розміщується в корені репозиторію
```

Цільова структура всього репозиторію:

```text
pampilo_bot/
  register-service/
    api/
    client/
  trading-service/ або адаптована структура поточних ботів
  infra/
    compose/
      docker-compose.infra.yaml
      docker-compose.apps.yaml
      docker-compose.dev.yaml
      docker-compose.prod.yaml
      docker-compose.mailpit.yaml
      docker-compose.observability.yaml
    gateway/
    observability/
    scripts/
  docker-compose.yaml
  Makefile
  package.json або інший root task runner
```

Фізичне перейменування модулів можна виконати після функціонального очищення. На першому етапі допустимо зберегти наявні назви `module-auth`, `module-auth-token`, `module-user`, `module-account` і `module-mail`, щоб зменшити обсяг одночасних змін.

## 4. Що необхідно залишити

### 4.1. API-модулі

Залишити та адаптувати:

| Модуль | Причина | Необхідна переробка |
|---|---|---|
| `module-auth` | guards, decorators, login orchestration, JWT strategies | замінити athlete/coach ролі на neutral customer roles |
| `module-auth-token` | access/refresh tokens, sessions, checkpoints, devices, reauth, security events | зберегти; відокремити від gym user payload |
| `module-customer-auth` | public registration/login API | перетворити на `identity-auth`; створити один generic register endpoint |
| `module-admin-auth` | адміністративний realm | залишити для platform admin, прибрати marketplace-specific permissions |
| `module-user` | user persistence, password hashing, uniqueness, cleanup | спростити user model до identity-полів |
| `module-account` | password reset, email change, sessions і security settings | видалити gym personalization/setup state, залишити identity security flows |
| `module-mail` | SMTP, templates, outbox, retry, readiness | залишити core delivery; спростити admin tooling за потреби |
| `module-encrypt` | захист токенів і mail secrets | залишити |
| `module-drizzle` | PostgreSQL і транзакції | залишити, очистити schema aggregator |
| `module-logger` | redaction, request id, audit-safe logging | залишити |
| `common/redis` | token revocation, rate limits, mail queue | залишити |
| `common/rate-limiting` | захист registration/email flows | залишити й переглянути policy |
| `common/health` | liveness/readiness | залишити й спростити dependency checks |
| `config` | environment validation, JWT, Redis, mail, DB | залишити лише потрібні конфігурації |

### 4.2. Email-функціональність

Залишити:

- SMTP delivery;
- transactional outbox;
- BullMQ worker, retries і backoff;
- verification-code template;
- password-reset template;
- email-change-confirmation template;
- security-alert template;
- mail audit events;
- readiness і базові delivery metrics;
- Mailpit для локальної розробки;
- encryption mail settings.

Admin mail preview, test-email endpoints і runtime SMTP settings можна залишити лише для `PLATFORM_ADMIN`. Якщо конфігурація SMTP керуватиметься виключно через secret manager/environment, ці admin endpoints і client UI потрібно видалити після перенесення конфігурації.

### 4.3. Client

Залишити та адаптувати:

- auth layout;
- login page/form;
- generic registration page/form;
- checkpoint approval;
- logout;
- refresh-token client/server logic;
- proxy/auth-cookie handling;
- current-session resolver;
- reauthentication modal;
- сторінки password reset/email verification, які потрібно виділити з account flows або додати;
- security account page: password, email, sessions, known devices, account deletion;
- системні error/loading states;
- мінімальні reusable UI components і theme assets, які реально імпортує auth UI.

### 4.4. Інфраструктура

Залишити:

- PostgreSQL;
- Redis;
- Mailpit у development;
- API container;
- client container, якщо окремий UI реєстрації є частиною сервісу;
- Nginx gateway;
- Loki/Alloy/Grafana або іншу погоджену observability-систему;
- Docker/Makefile scripts після очищення profiles і назв сервісів.

Інфраструктура має бути спільною для всього репозиторію. Потрібні файли з `register-service/infra` слід винести в кореневу `infra/`, а root compose і task-runner зробити єдиними точками запуску register-service, trading service та спільних залежностей.

## 5. Що потрібно видалити

### 5.1. API gym/marketplace домени

Після розв'язання залежностей видалити:

```text
api/src/module-athlete-account
api/src/module-athlete-frontend
api/src/module-coach-account
api/src/module-coach-frontend
api/src/module-trainer
api/src/module-sport
api/src/module-program
api/src/module-chat
api/src/module-review
api/src/module-feedback
api/src/module-moderation
api/src/module-profile
api/src/module-media
api/src/module-object-storage
api/src/module-translation
```

`module-admin-frontend` не слід видаляти механічно: спочатку винести потрібні identity/mail admin endpoints. Після цього marketplace admin shell видалити або замінити мінімальним platform admin module.

`module-data-patch` можна видалити, якщо після очищення не використовується як загальний механізм production data patches. Якщо механізм потрібен, залишити infrastructure core, але видалити gym/storage patches.

### 5.2. Client marketplace features

Видалити після перевірки import graph:

```text
client/src/features/module-chat
client/src/features/module-feedback
client/src/features/module-media
client/src/features/module-profile
client/src/features/module-program
client/src/features/module-review
client/src/features/module-sport
client/src/features/module-trainer
client/src/features/module-admin-storage
```

`module-account` не видаляти повністю. З нього потрібно прибрати:

- `athlete-personalization-form`;
- `coach-setup-form`;
- profile/dashboard personalization;
- sport-specific account settings.

Залишити security settings, sessions, password/email changes, account deactivation і deletion.

`module-menu`, `module-shell` і `module-admin-shell` спростити до мінімального auth/account UI або видалити, якщо register-service буде headless API без власного кабінету.

### 5.3. Client routes

Видалити:

```text
client/src/app/(customer)/athlete
client/src/app/(customer)/coach
client/src/app/(customer)/contacts
client/src/app/(customer)/market
client/src/app/admin/(authenticated)/settings/storage
```

Замість athlete/coach account routes створити нейтральний `/account/security`.

Admin dashboard залишити як затверджений operational UI для platform admin. Поточний admin shell не видаляти далі: `/admin/dashboard`, `/admin/settings/mail`, `/admin/settings/payment` і sidebar у стилі старого UI залишаються частиною register-service. Marketplace dashboards і storage settings видалити; `/admin/settings/storage` не повертати.

### 5.4. Інфраструктура та залежності

Видалити після видалення object storage/media/profile photo:

- MinIO і `minio-init`;
- fake GCS і `local-gcs-init`;
- object-storage volumes;
- S3/GCS environment variables;
- object-storage data patches;
- dashboards, що містять лише marketplace/media метрики.

Після підтвердження відсутності імпортів видалити з API dependencies:

- `@aws-sdk/client-s3`;
- `@aws-sdk/s3-request-presigner`;
- `@google-cloud/storage`;
- `sharp`;
- `socket.io` і `@nestjs/websockets`, якщо вони використовуються тільки chat;
- `@socket.io/redis-adapter`;
- `@nestjs/cqrs`, якщо після очищення він не використовується.

Після очищення client видалити невикористані UI/chart/builder dependencies, зокрема DnD, charts, KlineCharts, date pickers і virtualization, але лише за результатом import/dependency audit.

Після перенесення спільної інфраструктури видалити `register-service/infra`. Перед видаленням потрібно переконатися, що всі актуальні конфігурації перенесені, відносні build contexts виправлені, а root-команди повністю замінюють локальні команди register-service.

### 5.5. Консолідація `register-service/infra` у корені

Поточна `register-service/infra` описує не лише register-service, а й спільні PostgreSQL, Redis, mail, gateway та observability ресурси. Щоб репозиторій був цілісним, infrastructure ownership потрібно підняти на root-рівень.

Перенести й адаптувати:

```text
register-service/infra/compose                 -> infra/compose
register-service/infra/nginx                   -> infra/gateway або infra/nginx
register-service/infra/observability           -> infra/observability
register-service/infra/scripts                 -> infra/scripts
register-service/docker-compose.yaml           -> root docker-compose.yaml / compose fragments
register-service/Makefile                      -> root Makefile
register-service/package.json                  -> root task runner, якщо він потрібен
```

Під час перенесення:

- не перезаписувати наосліп наявні root `docker-compose.yaml` і `docker-compose-db.yaml`;
- спочатку скласти матрицю всіх сервісів, volumes, networks, ports і environment variables;
- об'єднати конфігурації в один root compose stack із профілями;
- виправити `build.context`, `dockerfile`, volume paths та env-file paths відносно кореня;
- перейменувати `pampilo_api`, `pampilo_client`, `pampilo_db` та інші gym-oriented container names;
- визначити одну root network для внутрішньої взаємодії сервісів;
- уникати жорстких `container_name`, якщо потрібне горизонтальне масштабування;
- розділити databases/users/schemas register-service і trading service, навіть якщо вони працюють на одному PostgreSQL instance;
- залишити один Redis instance лише якщо ізоляція key prefixes, ACL і queue names достатня; інакше розділити logical DB або instances;
- зробити gateway маршрути явними: identity/register API, register client і trading API;
- оновити healthcheck dependencies та порядок запуску;
- прибрати MinIO/GCS та їх volumes після видалення media/object-storage коду;
- перейменувати Loki/Grafana dashboards і labels відповідно до нових назв сервісів;
- винести production secrets із compose у secret manager або deployment environment;
- додати root-команди для `config`, `up`, `down`, `logs`, `ps`, migrations і tests.

Рекомендовані root compose profiles:

```text
infra          # PostgreSQL, Redis
identity       # register API/client
trading        # market data, strategy та execution workers
mail           # Mailpit лише для development
gateway        # reverse proxy
observability  # Loki, Alloy, Grafana
```

Очікуваний результат:

- проєкт запускається з кореня однією узгодженою командою;
- `register-service` не містить власної копії загальної інфраструктури;
- усі сервіси використовують єдині conventions для networks, logs, health та environment;
- development і production topology описані централізовано;
- окремі application services можна збирати й масштабувати незалежно.

## 6. Обов'язкові refactoring-кроки перед видаленням

### 6.1. Нейтральна модель ролей

Поточні ролі:

```text
ATHLETE | COACH | ADMIN | SUPER_ADMIN
```

Замінити на модель на кшталт:

```text
USER | PLATFORM_ADMIN | SUPER_ADMIN
```

Якщо trading product матиме організації, tenant roles (`OWNER`, `ADMIN`, `MEMBER`) краще зберігати окремо від platform roles.

Міграція повинна оновити JWT payload, guards, cookies/auth realm, DTO, client enums і тести. Роль не можна приймати з public registration DTO: новий користувач завжди отримує безпечну default role на сервері.

### 6.2. Generic registration

Замінити:

```text
POST /register/athlete
POST /register/coach
```

на:

```text
POST /auth/register
```

Generic DTO повинен містити мінімум:

```text
email
password
displayName або name
acceptedTermsVersion
acceptedRiskDisclosureVersion
```

`CustomerRegistrationRepository` повинен створювати лише user/tenant/membership та verification state. Він не повинен імпортувати athlete/coach schemas.

### 6.3. Email verification

Поточний login checkpoint не слід автоматично вважати повною email verification. Потрібно явно визначити:

- чи email підтверджується до першого login;
- TTL verification token/code;
- resend policy;
- rate limits;
- одноразовість token;
- стан `email_verified_at`;
- блокування trading onboarding до verification.

### 6.4. Account recovery API

Password reset/email change зараз доступні через athlete/coach controllers. Перед їх видаленням створити нейтральний controller, наприклад:

```text
POST /auth/password/forgot
POST /auth/password/reset
POST /account/password/change
POST /account/email/change/request
POST /account/email/change/confirm
GET  /account/sessions
DELETE /account/sessions/:id
DELETE /account
```

Сервіси й repositories із `module-account`, потрібні цим endpoints, залишити. `account-setup-state` та gym personalization видалити після перенесення recovery/security flows.

### 6.5. User schema

Залишити лише identity-дані:

- `id`;
- normalized email;
- password hash;
- display name, якщо він потрібен;
- platform status;
- platform role;
- `email_verified_at`;
- pending email change;
- timestamps;
- deletion/anonymization state.

Sport/profile поля мають бути видалені. Пароль ніколи не повертається через API і не потрапляє в audit/log payload.

### 6.6. Trading-service integration

Додати незалежно від gym-модулів:

- stable `user_id` і `tenant_id`;
- endpoint або signed token для переходу до trading onboarding;
- JWKS або інший механізм перевірки підпису токенів;
- domain events `UserRegistered`, `EmailVerified`, `TradingAccessGranted`, `TradingAccessSuspended`, `UserDisabled`, `UserDeleted`;
- transactional outbox;
- idempotency/event versioning;
- service-to-service authorization.

Registration service не зберігає API keys бірж і не виконує торгові операції.

## 7. База даних і міграції

Поточні Drizzle migrations містять marketplace та object-storage таблиці. Не рекомендується просто видалити окремі SQL-файли зі старого migration chain.

Рекомендований порядок:

1. Зафіксувати перелік identity-таблиць, які залишаються.
2. Створити новий чистий baseline schema для нового сервісу.
3. Згенерувати новий migration chain для порожньої identity database.
4. Окремо підготувати data migration, лише якщо потрібно переносити реальних користувачів.
5. Не переносити gym profiles, programs, media metadata або object-storage settings.
6. Оновити `module-drizzle/schemas/index.ts`, залишивши тільки identity, auth-token, account-security, mail, integration outbox/inbox schemas.
7. Додати foreign keys, unique normalized-email index, TTL/cleanup indexes для tokens, challenges, sessions та outbox.

Мінімальні таблиці:

```text
users
tenants
memberships
sessions
tokens або refresh_tokens
known_devices
login_challenges
reauth_confirmations
password_reset_challenges
email_change_challenges
security_events
mail_outbox
mail_audit_events
mail_settings (тільки якщо runtime configuration потрібна)
integration_outbox
processed_inbox_events (якщо сервіс приймає зовнішні події)
```

## 8. AppModule після очищення

Цільовий `AppModule` повинен імпортувати приблизно такі модулі:

```text
ConfigModule
ScheduleModule
BullModule
DrizzleModule
RedisModule (global або через consumers)
UserModule
AuthTokenModule
AuthModule
IdentityAuthModule
AdminAuthModule
AccountSecurityModule
MailModule
IntegrationModule
LoggerModule
HealthModule
```

Видалити imports `AthleteFrontendModule`, `CoachFrontendModule`, `SportModule`, `TrainerModule`, `TranslationModule`, `ModerationModule`, object storage та profile photo modules.

## 9. Тести

### 9.1. Залишити й адаптувати

- `module-auth`;
- `module-auth-token`;
- `module-customer-auth`, перейменувавши на generic identity auth;
- `module-admin-auth`;
- `module-user`;
- identity-частину `module-account`;
- `module-mail`;
- `module-encrypt`;
- `module-logger`;
- common health/rate-limit/config tests;
- client `module-auth`;
- client security-частину `module-account`;
- proxy, cookies, refresh і session resolver tests.

### 9.2. Видалити

Тести chat, feedback, media, moderation, object storage, profile, program, review, sport, trainer, athlete/coach setup та marketplace menu/routes.

### 9.3. Додати

- generic registration happy path;
- duplicate/normalized email;
- server-controlled default role;
- email verification і resend limits;
- login, logout і token rotation;
- refresh-token reuse/revocation;
- password reset та email change;
- session/device revocation;
- tenant isolation;
- account disabled/deleted behavior;
- mail outbox retry та idempotency;
- signed trading-onboarding token;
- domain outbox events;
- security log redaction;
- Redis/mail/database outage behavior.

## 10. Фази реалізації

Фази виконуються послідовно. Наступна фаза починається тільки після проходження exit criteria попередньої. Видалення коду завжди відбувається після створення, переключення та перевірки його нейтральної заміни.

### Фаза 0. Зафіксувати вихідний стан

Роботи:

- зафіксувати версії Node.js, npm і зовнішніх сервісів;
- виконати API/client build, unit tests і compose config;
- зафіксувати відомі помилки окремо від регресій очищення;
- зберегти перелік runtime endpoints і background jobs.

Результат: існує відтворюваний baseline, з яким порівнюються наступні фази.

Exit criteria:

- команди перевірки задокументовані;
- відомо, які тести проходять до початку змін;
- жодний модуль ще не видалений.

### Фаза 1. Побудувати карту залежностей

Роботи:

- побудувати import graph API і client;
- скласти перелік NestJS modules, controllers, providers і scheduled jobs;
- зафіксувати Drizzle schemas, migrations, data patches і foreign keys;
- скласти матрицю npm dependencies та їх consumers;
- позначити кожен компонент як `KEEP`, `REFACTOR`, `MOVE` або `DELETE`.

Результат: кожне майбутнє видалення має відомі вхідні й вихідні залежності.

Exit criteria:

- немає непроаналізованих imports у auth/user/account/mail;
- password reset, email change, sessions і mail dependencies нанесені на карту.

### Фаза 2. Зафіксувати цільові identity-контракти

Роботи:

- затвердити generic user, tenant і membership models;
- затвердити platform roles і tenant roles;
- визначити registration, login, refresh, logout та account-security API;
- визначити email verification policy;
- визначити token/cookie policy та auth realms;
- зафіксувати integration events для trading service.

Результат: реалізація має стабільну ціль і не переносить gym-модель у нові модулі.

Exit criteria:

- public registration не приймає privileged role;
- `ATHLETE` і `COACH` відсутні в цільових контрактах;
- API та event contracts версіоновані.

### Фаза 3. Підготувати root infrastructure scaffold

Роботи:

- інвентаризувати root compose і `register-service/infra`;
- створити root `infra/` та початкові compose fragments без видалення старих файлів;
- визначити profiles `infra`, `identity`, `trading`, `mail`, `gateway`, `observability`;
- погодити networks, ports, database boundaries і Redis queue/key prefixes;
- додати root task-runner commands для config/build/test.

Результат: нова root-структура існує паралельно зі старою та ще не є єдиним production path.

Exit criteria:

- root compose проходить syntactic config validation;
- старий dev flow продовжує працювати;
- жодний volume або production data path не видалений.

### Фаза 4. Ввести нейтральні ролі з compatibility layer

Роботи:

- додати `USER`, `PLATFORM_ADMIN`, `SUPER_ADMIN`;
- відокремити platform roles від tenant membership roles;
- оновити JWT payload і authorization guards;
- тимчасово додати контрольований mapping старих athlete/coach users у `USER`;
- додати migration і compatibility tests.

Результат: нова identity-логіка використовує нейтральні ролі, а старі endpoints тимчасово не зламані.

Exit criteria:

- нові tokens не потребують athlete/coach role;
- admin role неможливо отримати через public DTO;
- старі auth regression tests або їх compatibility-аналоги проходять.

### Фаза 5. Спростити user persistence

Роботи:

- створити нейтральний user repository;
- додати normalized email і безпечний unique index;
- додати `email_verified_at`, account status і deletion state;
- винести tenant/membership creation в окрему транзакційну операцію;
- не видаляти athlete/coach schemas на цій фазі.

Результат: generic user можна створити без спортивного профілю.

Exit criteria:

- repository tests доводять створення user/tenant/membership однією транзакцією;
- password hash не повертається через API;
- duplicate email обробляється детерміновано.

### Фаза 6. Реалізувати generic registration API

Роботи:

- додати `POST /auth/register`;
- створити generic DTO без gender, date of birth, specialization і languages;
- призначати `USER` тільки на сервері;
- переключити registration transaction на neutral user persistence;
- залишити athlete/coach endpoints тимчасово deprecated, але не використовувати їх у новому flow.

Результат: нова реєстрація не імпортує athlete/coach repositories або schemas.

Exit criteria:

- generic registration має unit та integration tests;
- реєстрація не створює gym tables;
- повторний запит не створює неконсистентні user/tenant records.

### Фаза 7. Реалізувати підтвердження email

Роботи:

- додати verification challenge/token schema;
- створити verify і resend endpoints;
- підключити verification email template через mail outbox;
- додати TTL, одноразовість, resend cooldown і rate limits;
- блокувати trading onboarding до verification.

Результат: реєстрація та підтвердження email є окремими явними станами.

Exit criteria:

- token не можна використати повторно;
- resend не обходить rate limit;
- успішне підтвердження записує `email_verified_at` і audit event.

### Фаза 8. Нейтралізувати login і session flow

Роботи:

- перевести login, refresh, logout і `me` на generic user;
- зберегти token rotation, revocation, sessions і known devices;
- адаптувати checkpoints і risk policy;
- прибрати athlete/coach role selection із login DTO;
- оновити cookie names лише через контрольовану compatibility migration.

Результат: повний auth lifecycle працює без gym-ролей.

Exit criteria:

- register -> verify -> login -> refresh -> logout проходить end-to-end;
- session revocation працює;
- refresh-token reuse виявляється та блокується.

### Фаза 9. Винести account recovery у нейтральний API

Роботи:

- створити neutral account-security controller;
- перенести password forgot/reset/change;
- перенести email change request/confirm;
- перенести sessions/device listing і revocation;
- перенести account deactivate/delete;
- не видаляти athlete/coach controllers, доки нові endpoints не перевірені.

Результат: identity recovery більше не залежить від athlete/coach route modules.

Exit criteria:

- усі recovery flows мають unit та integration tests;
- email change і password reset відкликають потрібні sessions;
- athlete/coach controllers більше не є єдиним шляхом до identity-функцій.

### Фаза 10. Очистити mail core

Роботи:

- залишити SMTP, outbox, BullMQ, retries, audit і readiness;
- залишити тільки identity/security templates;
- визначити долю runtime mail settings та admin preview;
- видалити marketplace mail templates і recipients;
- перевірити redaction адрес і delivery payload.

Результат: mail module обслуговує лише registration, recovery та security flows.

Exit criteria:

- verification, reset, email-change і security-alert emails проходять через outbox;
- retry не створює дубльовану бізнес-подію;
- Mailpit development flow працює.

### Статус реалізації фаз 0–10

Станом на 11 липня 2026 року фази 0–10 реалізовані та пройшли production-readiness gate:

- зафіксовано baseline, dependency map і versioned identity contracts;
- root compose використовує узгоджені профілі та проходить `config --quiet`;
- migration `0008_harden_identity_core.sql` застосована і перевірена на PostgreSQL;
- generic registration, email verification, auth/session/device та neutral account-security flows покриті тестами;
- верифікація email є транзакційною, одноразовою та створює security audit event;
- mail flow перевірено реальним проходженням `register -> outbox -> BullMQ -> SMTP -> Mailpit`;
- API lint/build і 176 test suites (1354 tests), client build і 58 test files (297 tests) проходять.

`EmailVerifiedGuard` є обов'язковим integration guard для майбутнього trading onboarding; сам trading endpoint створюється у відповідній наступній фазі й не входить у фази 0–10.

### Фаза 11. Переробити client auth flow

Роботи:

- замінити athlete/coach registration steps на generic form;
- додати verify-email, forgot-password і reset-password pages;
- адаптувати login, checkpoint, refresh і proxy logic;
- створити neutral `/account/security`;
- залишити старі marketplace routes тимчасово недоступними через нову навігацію, але ще не видаляти їх код.

Результат: користувач проходить весь identity flow через новий UI.

Exit criteria:

- client auth tests проходять;
- UI не запитує gym-поля;
- account security не імпортує profile/program/sport features.

### Фаза 12. Додати інтеграцію з trading service

Роботи:

- додати stable `user_id`, `tenant_id` і membership context;
- реалізувати signed onboarding token або JWKS validation contract;
- створити transactional identity outbox;
- публікувати `UserRegistered`, `EmailVerified`, `TradingAccessGranted`, `UserDisabled`, `UserDeleted`;
- додати service-to-service authorization та idempotency.

Результат: trading service отримує identity context без прямого доступу до identity database.

Exit criteria:

- події відновлюються після рестарту publisher;
- повторна доставка безпечна;
- onboarding token короткоживучий, scoped і перевіряється trading service contract tests.

### Фаза 13. Від'єднати athlete/coach код від runtime

Роботи:

- прибрати athlete/coach modules з `AppModule` і client navigation;
- вимкнути deprecated registration endpoints;
- прибрати athlete/coach schemas із generic registration та account security;
- виконати import/cycle scan;
- ще не видаляти файли до успішної runtime перевірки.

Результат: gym account modules фізично існують, але не входять у production runtime.

Exit criteria:

- production module graph не містить athlete/coach modules;
- generic identity flows проходять без gym tables;
- deprecated endpoints повертають погоджений статус або повністю зняті з routing.

### Фаза 14. Видалити малі leaf-домени

Роботи:

- окремими малими змінами видалити feedback;
- потім review;
- потім sport;
- потім trainer;
- потім moderation;
- потім translation, якщо після очищення identity UI/API вона не використовується;
- після кожного видалення очистити imports, tests і package usage.

Результат: видалено домени з найменшим dependency radius.

Exit criteria:

- після кожного домену окремо проходять API/client build і identity tests;
- Drizzle runtime schema не імпортує видалену схему.

### Фаза 15. Видалити program і chat

Роботи:

- спочатку видалити program frontend/API і його tests;
- окремою зміною видалити chat frontend/API, gateways і jobs;
- після chat видалити Socket.IO/WebSocket dependencies, якщо більше немає consumers;
- оновити Redis config, прибравши chat-specific keys/channels.

Результат: realtime і program marketplace logic відсутні.

Exit criteria:

- Redis використовується лише auth/rate-limit/mail/integration flows;
- dependency audit підтверджує відсутність Socket.IO consumers.

### Фаза 16. Видалити media, profile та object storage

Роботи:

- від'єднати profile photo від generic user;
- видалити media і profile modules;
- видалити object-storage modules і data patches;
- видалити S3/GCS/Sharp dependencies після import scan;
- поки не видаляти MinIO/GCS compose services до переключення root infra.

Результат: application code більше не залежить від object storage.

Exit criteria:

- API/client build не потребує AWS, GCS або Sharp packages;
- user model не містить обов'язкового profile/media relation.

### Фаза 17. Видалити athlete/coach файли та marketplace shell

Роботи:

- видалити athlete account/frontend;
- окремо видалити coach account/frontend;
- видалити marketplace menu, dashboards і непотрібний admin shell;
- залишити затверджений platform admin UI у поточному стані: `/admin/dashboard`, `/admin/settings/mail`, `/admin/settings/payment` та admin sidebar зі старого UI; ці сторінки й sidebar більше не видаляти в межах cleanup;
- не повертати і не переносити storage/file-storage UI (`/admin/settings/storage`, `module-admin-storage`);
- видалити відповідні tests і fixtures.

Результат: gym marketplace domain фізично відсутній у вихідному коді.

Exit criteria:

- `rg`/import graph не знаходить runtime references на athlete, coach або gym domain;
- identity UI та API проходять повний regression suite.

### Фаза 18. Створити чистий DB migration baseline

Поточне рішення: baseline зафіксовано в `api/drizzle-migrations/0000_happy_azazel.sql`, Drizzle config використовує явний aggregator `api/src/module-drizzle/schemas/index.ts`, а перелік таблиць, migration-from-zero procedure, data migration policy та rollback/recovery strategy задокументовані в `api/docs/database-baseline.md`.

Роботи:

- зафіксувати фінальний перелік identity tables;
- очистити Drizzle schema aggregator;
- створити migration-from-zero для нової identity database;
- підготувати окремий migration tool для існуючих users, якщо перенос даних потрібен;
- видалити gym/media/object-storage migrations і data patches лише з нового baseline chain.

Результат: нове середовище створює лише identity/mail/integration tables.

Exit criteria:

- migration-from-zero проходить на порожній БД;
- upgrade test проходить для підтримуваного джерела даних;
- rollback/recovery strategy задокументована.

### Фаза 19. Очистити package dependencies і lockfiles

Поточне рішення: API/client manifests очищені від пакетів без consumers, lockfiles створені через `npm install`/`npm audit fix`, clean install/build/tests проходять. Залишкові `npm audit` findings не виправлялись через `--force`, бо npm пропонує breaking downgrade/force path для `drizzle-kit`/`next` transitive advisories. License inventory виконаний; залишкові non-permissive/unknown licenses є транзитивними пакетами, не прямими application dependencies.

Роботи:

- видаляти лише packages без consumers;
- очистити API dependencies;
- окремо очистити client dependencies;
- штатно оновити lockfiles;
- виконати license та vulnerability audit.

Результат: manifests відповідають реальному identity runtime.

Exit criteria:

- clean install, build і tests проходять;
- import graph не має unresolved modules;
- production image не містить object-storage/chat libraries.

### Фаза 20. Переключити запуск на root infrastructure

Поточне рішення: root `infra/compose` є основним шляхом запуску для identity, trading і observability. Root `Makefile` та `register-service/package.json` переключені на root compose/task runner. Gateway має health routes `/health/identity` і `/health/trading`. Dev stack підтримує параметризовані host-порти для паралельної перевірки поруч зі старим стеком. Trading bot запускається в dev через `paper` mode, production бере execution mode з `spot_grid_bot/.env.production`.

Роботи:

- перенести й адаптувати compose fragments, gateway, scripts та observability;
- виправити build contexts, env paths, networks, volumes і health checks;
- додати identity та trading routes у gateway;
- переключити development commands на root task runner;
- виконати паралельне порівняння старого і нового stack.

Результат: root orchestration є основним шляхом запуску, але старі infra-файли ще збережені для контрольної перевірки.

Exit criteria:

- root dev stack запускається з нуля;
- identity і trading health checks доступні через gateway;
- production compose config валідний;
- дані Postgres/Redis не втрачені під час переключення.

### Фаза 21. Видалити локальну `register-service/infra`

Поточне рішення: локальні дублікати `register-service/infra`, `register-service/docker-compose.yaml` і `register-service/Makefile` видалені. Root compose використовує project name `pampilo-platform`, project-scoped container names, `platform_network` і `platform_*` volumes. MinIO/fake GCS services у root compose відсутні, object-storage env залишки прибрані з dev/prod env. Observability discovery оновлений під нові compose container names. Root backup/restore smoke перевірений на нових `platform_postgres_data`/`platform_redis_data`: PostgreSQL custom dump відновлюється в окрему probe DB, Redis RDB snapshot створюється. Старі `postgres_data_pampilo`/`redis_data` не видалялись; повний `pg_dump` старого running DB volume окремо блокується orphan trading candle objects із відсутнім schema OID і потребує окремого catalog cleanup перед міграцією.

Роботи:

- видалити MinIO/fake GCS services і volumes із root config;
- видалити `register-service/infra`;
- видалити локальні дублікати compose, Makefile і root package scripts;
- перейменувати `pampilo_*` containers, networks, labels і dashboards;
- перевірити observability для нових назв сервісів.

Результат: у репозиторії існує одна централізована infrastructure topology.

Exit criteria:

- пошук не знаходить активних посилань на старі infra paths;
- root dev і production config проходять повторну перевірку;
- backup/restore volumes перевірено.

### Фаза 22. Security hardening

Поточне рішення: bootstrap отримав browser security headers і CSRF origin check для cookie-auth unsafe requests; у production `CSRF_ORIGIN_CHECK_ENABLED` вмикається за замовчуванням. JWT rotation підтримується через `JWT_PREVIOUS_SECRETS`: нові токени підписуються `JWT_SECRET`, а verify/Passport приймає поточний і попередні secrets. Log redaction розширений на case/underscore/hyphen variants (`Authorization`, `api_key`, `set-cookie`, session/service tokens). Redis/session revocation fail-closed, SMTP/mail readiness fallback, PostgreSQL outbox failure handling, account deletion cascade і token revocation перевірені unit tests. `.env`/`.env.production` залишаються локальними ignored файлами; secrets не bake-яться в images. Dependency scan виконаний: залишкові advisories у `drizzle-kit`/`esbuild` і `next`/`postcss` не виправлені через `--force`, бо npm пропонує breaking downgrade/force path.

Роботи:

- провести review JWT, cookies, CORS, CSRF, session fixation і rate limits;
- перевірити log redaction і security events;
- додати secret rotation;
- виконати tests Redis, SMTP і PostgreSQL outage;
- перевірити account deletion та token revocation;
- провести dependency/security scan.

Результат: identity service готовий до передproductionного security review.

Exit criteria:

- критичні security findings закриті;
- secrets відсутні в repository, images і logs;
- failure modes мають fail-safe поведінку.

### Фаза 23. Навантажувальна та операційна перевірка

Поточне рішення: додано операційний пакет `register-service/ops` із SLO/alert thresholds, runbooks для DB/Redis/SMTP/integration outage, dependency-free load smoke runner і backup/restore smoke script. Root `Makefile` отримав `ops-load-smoke` та `ops-backup-restore-smoke`. Grafana/Loki provisioning перейменовано під platform operations (`platform-loki`, `platform-operations`), а dashboard очищено від старих marketplace channel queries і доповнено audit/security search та mail delivery panels. `/metrics` переведено у scrape-ready plain text без глобального JSON response envelope. Recovery exercise виконаний на root compose stack: PostgreSQL custom dump відновлено в окрему probe DB, Redis RDB snapshot створено, API container перебудовано і перевірено через readiness/metrics. Load smoke проти Docker API (`concurrency=4`, `duration=10s`) завершився з `1363` requests, `0` failed, max latency `60ms`; сценарії покрили readiness, metrics, invalid login, invalid register/rate-limit, refresh без cookie і password reset probe. Post-smoke метрики: `mail_outbox_lag_seconds=0`, `bullmq_queue_depth=0`, `smtp_delivery_success_rate=1.0000`, SMTP error counters `0`.

Роботи:

- навантажити register/login/refresh/mail flows;
- перевірити BullMQ backlog і mail retries;
- протестувати backup/restore identity database;
- створити runbooks для DB, Redis, SMTP і integration outage;
- перевірити dashboards, alerts і audit search.

Результат: визначені capacity limits і операційні процедури.

Exit criteria:

- SLO та alert thresholds задокументовані;
- recovery exercises завершені;
- система витримує погоджене beta-навантаження.

### Фаза 24. Фінальна acceptance-перевірка

Поточне рішення: виконано фінальну acceptance-перевірку очищеного register-service. Package names перейменовано на `register-service-api` і `register-service-client`, прибрано останній runtime текст `Gym Marketplace` з email template. Виправлено production-blockers, знайдені acceptance flow: `MAIL_SETTINGS_ENCRYPTION_KEY` додано в dev/prod env для mail outbox encryption; root compose тепер явно прокидає `DB_USER`, `DB_PASSWORD` і `DB_NAME` в API container, щоб staging/acceptance могли запускати API на окремій БД; `TradingOnboardingController` запускає JWT/role guard перед `EmailVerifiedGuard`. Migration-from-zero виконано на порожній `phase24_zero` DB: створені тільки identity/account/mail/integration таблиці. Runtime E2E пройшов: `POST /auth/register` створює generic `role=user`, privileged `role=SUPER_ADMIN` payload відхиляється `400`, створюються `email_verifications`, `mail_outbox`, `identity_outbox_events` і primary `memberships`; `POST /integration/trading/onboarding-token` до email verification повертає `403`, після verification повертає Bearer token з `aud=trading-service`. Release checklist і rollback plan додані в `register-service/ops`.

Роботи:

- перевірити всі критерії розділу 12;
- виконати clean install, migration-from-zero, build і повний test suite;
- виконати end-to-end registration-to-trading-onboarding flow;
- підтвердити відсутність gym marketplace коду та інфраструктурних дублікатів;
- підготувати release checklist і rollback plan.

Результат: очищення завершено й register-service готовий до окремого релізу.

Exit criteria:

- усі критерії завершення виконані;
- немає невирішених critical/high defects;
- release і rollback перевірені на staging.

## 11. Правила безпечного очищення

- Не видаляти модуль тільки за назвою; спочатку перевіряти import graph і runtime wiring.
- Не змішувати функціональну переробку auth із масовим видаленням у одному коміті.
- Не редагувати старі production migrations без визначеної migration strategy.
- Не переносити Pampilo `.env`, production credentials або customer data.
- Не залишати public endpoint, який дозволяє обрати `ADMIN` або `SUPER_ADMIN` під час реєстрації.
- Не видаляти Redis/BullMQ, доки token revocation, rate limiting і mail outbox залежать від них.
- Не видаляти athlete/coach account modules, доки password reset/email change не доступні через neutral controller.
- Не видаляти mail admin tooling, доки не визначено інший спосіб конфігурації та діагностики SMTP.
- Не видаляти `register-service/infra`, доки root compose не відтворює весь потрібний development і production stack.
- Не перезаписувати чинні root compose-файли без порівняння сервісів, volumes, networks та даних PostgreSQL.
- Кожен етап має завершуватися зеленими build/tests і перевіркою compose config.

## 12. Критерії завершення

- У runtime і package names немає gym marketplace домену.
- Public registration створює generic user і не приймає privileged role.
- Немає imports athlete/coach/program/chat/media/profile/object-storage schemas.
- Працюють register, email verification, login, refresh, logout, checkpoints, password reset та email change.
- Працюють session/device management, security events і account deactivation/deletion.
- Системні email надсилаються через надійний outbox із retry та аудитом.
- API містить лише identity/account/mail/integration endpoints.
- Client містить лише auth, account security і за потреби platform admin UI.
- PostgreSQL schema не містить gym/marketplace таблиць.
- Compose не запускає MinIO/GCS або інші непотрібні сервіси.
- Уся спільна інфраструктура знаходиться в root `infra/`, а `register-service/infra` видалена.
- Register service і trading service запускаються з кореня через єдиний compose/task-runner.
- Root infrastructure має окремі service profiles та ізольовані persistence boundaries.
- Trading service може перевірити підписаний user/tenant context без доступу до identity database.
- Build, unit/integration tests, migration-from-zero та production compose проходять успішно.
