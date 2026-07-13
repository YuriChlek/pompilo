# Identity Contracts v1

Версія контракту: `identity.v1`.

## Моделі

### User

```text
id: UUID
email: normalized string
name: string
platformRole: USER | PLATFORM_ADMIN | SUPER_ADMIN
status: ACTIVE | DEACTIVATED | PENDING_DELETION | DELETED
emailVerifiedAt: timestamp | null
createdAt: timestamp
updatedAt: timestamp
```

### Tenant і membership

```text
Tenant: id, name, status, timestamps
Membership: userId, tenantId, role OWNER | ADMIN | MEMBER, timestamps
```

Platform role і tenant role є різними authorization dimensions.

## HTTP API v1

Public:

```text
POST /auth/register
POST /auth/login
POST /auth/refresh
POST /auth/logout
POST /auth/email/verify
POST /auth/email/resend
POST /auth/checkpoint/verify
POST /auth/checkpoint/resend
POST /auth/password/forgot
POST /auth/password/reset
```

Authenticated:

```text
GET    /auth/me
GET    /account/sessions
DELETE /account/sessions
DELETE /account/sessions/others
DELETE /account/sessions/:id
GET    /account/devices
DELETE /account/devices/:id
POST   /account/password/change
POST   /account/email/change/request
POST   /account/email/change/confirm
POST   /account/re-auth
POST   /account/deactivate
DELETE /account
```

Legacy unversioned routes можуть тимчасово існувати лише як compatibility aliases і мають бути позначені deprecated.

## Email verification policy

- token entropy: не менше 256 bits;
- у БД зберігається лише hash;
- TTL: 30 хвилин;
- token одноразовий;
- resend cooldown: 60 секунд плюс глобальний email-flow rate limit;
- update `email_verified_at`, consume token і audit event виконуються атомарно;
- trading onboarding вимагає verified email.

## Token/cookie policy

- access і refresh cookies: HttpOnly, Secure у production, SameSite=Lax, path `/`;
- refresh rotation і reuse detection є обов'язковими;
- password/email security changes відкликають активні sessions;
- customer та platform-admin realms використовують різні cookie names;
- legacy athlete/coach cookies тільки очищуються, нові не видаються.

## Integration events v1

Envelope:

```text
eventId: UUID
eventType: identity.v1.<EventName>
eventVersion: 1
occurredAt: ISO-8601 UTC
aggregateId: userId або tenantId
tenantId: UUID
payload: object
```

Події:

- `identity.v1.UserRegistered`;
- `identity.v1.EmailVerified`;
- `identity.v1.TradingAccessGranted`;
- `identity.v1.TradingAccessChanged`;
- `identity.v1.TradingAccessSuspended`;
- `identity.v1.UserDisabled`;
- `identity.v1.UserDeleted`.

Фактична transactional outbox реалізація цих подій належить фазі 12. До її завершення цей документ є source of truth для контракту.

## Security invariants

- public input ніколи не визначає privileged platform role;
- password hash, tokens і secrets не повертаються API;
- tenant context береться з підписаного identity context, не з довільного client field;
- trading service не читає identity database;
- біржові credentials не зберігаються identity service.

