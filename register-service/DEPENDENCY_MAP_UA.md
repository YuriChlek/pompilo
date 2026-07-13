# Register Service Dependency Map

## Класифікація

| Компонент | Рішення | Критичні залежності |
|---|---|---|
| `module-auth` | KEEP/REFACTOR | user, auth-token, mail, Redis |
| `module-auth-token` | KEEP/REFACTOR | user, Drizzle, Redis, encryption, mail |
| `module-customer-auth` | KEEP/REFACTOR | auth, user, auth-token, mail, transaction repository |
| `module-admin-auth` | KEEP/REFACTOR | auth, platform roles |
| `module-user` | KEEP/REFACTOR | Drizzle, auth roles, tenants/memberships |
| `module-account` | KEEP/REFACTOR | user, sessions, reauth, mail, Drizzle |
| `module-mail` | KEEP | PostgreSQL outbox, BullMQ, Redis, SMTP, encryption |
| `module-encrypt` | KEEP | environment encryption key |
| `module-drizzle` | KEEP/REFACTOR | PostgreSQL, schema aggregator |
| `module-logger` | KEEP | request context, redaction |
| `common/redis` | KEEP | tokens, rate limits, mail queue state |
| `common/health` | KEEP/REFACTOR | DB, Redis, mail readiness |
| athlete/coach account/frontend | MOVE THEN DELETE | account recovery must be neutral first |
| feedback/review/sport/trainer | DELETE | leaf marketplace domains |
| program/chat | DELETE | Redis/WebSocket and program domain |
| profile/media/object-storage | DELETE | S3/GCS/Sharp/MinIO |
| moderation/translation | DELETE if no identity consumer | marketplace content |
| `module-data-patch` | REFACTOR | keep mechanism only if identity patches remain |

## Identity dependency closure

```text
Customer/Admin Controllers
  -> AuthService / CustomerAuthService
    -> UserRepository + UserPasswordService
    -> AuthSessionService
      -> AuthTokenService
        -> PostgreSQL token/session repositories
        -> Redis deny list and rotation state
    -> KnownDeviceService + RiskPolicyService
    -> EmailVerificationService
      -> MailTemplateService

AccountSecurityController
  -> AccountSettingsService
    -> UserRepository
    -> SessionService + KnownDeviceService
    -> ReauthConfirmationService
    -> password/email challenge repositories
    -> MailTemplateService

MailTemplateService
  -> encrypted PostgreSQL outbox
    -> MailOutboxRelayService
      -> BullMQ stable jobId=idempotencyKey
        -> MailProcessorService
          -> SmtpMailService
```

## Schemas, які належать identity core

- users, tenants, memberships;
- tokens, sessions, known_devices;
- login_challenges, reauth_confirmations, security_events;
- email_verifications;
- password_reset_challenges, email_change_challenges;
- mail_outbox, mail_audit_events, mail_settings.

Gym/media schemas залишаються у старому migration chain до фази 18, але новий identity-код не повинен створювати на них нові залежності.

## Scheduled jobs

| Job | Рішення |
|---|---|
| token cleanup | KEEP |
| account challenge cleanup | KEEP |
| user deletion cleanup | KEEP |
| mail relay/processor/cleanup | KEEP |
| media upload cleanup | DELETE у media phase |
| chat presence processor | DELETE у chat phase |

## npm dependency ownership

- PostgreSQL/Drizzle, JWT/Passport, Argon2, Redis, BullMQ, Nodemailer, React Email, Pino: KEEP.
- AWS SDK, GCS, Sharp: DELETE після object-storage phase.
- Socket.IO/WebSockets: DELETE після chat phase.
- DnD, charts, KlineCharts, virtualization: DELETE після client marketplace cleanup.

## Правило оновлення карти

Перед видаленням будь-якого модуля необхідно повторити API/client import graph, перевірити Drizzle schema imports, Nest `AppModule`, tests, compose services та package consumers. Зміна з `KEEP/REFACTOR/MOVE` на `DELETE` виконується тільки після зеленої нейтральної заміни.

