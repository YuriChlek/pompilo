# Mail Module

This module provides transactional email delivery using a secure database outbox pattern, BullMQ background processor, and React Email templates.

## Identity Service Boundary

The mail module is retained as part of the Identity and Subscription Service. Its production
scope is limited to registration, authentication, account recovery, session security and
identity lifecycle notifications.

Allowed templates are:

- account email verification;
- login/checkpoint verification codes;
- password reset;
- email change confirmation;
- security alerts.

Marketplace, trainer, program, review, chat and media notification templates or
recipients must not be added to this module.

Runtime SMTP settings, test delivery and template preview endpoints are intentionally retained
for operational support. They are restricted to `PLATFORM_ADMIN` and `SUPER_ADMIN`.

The `/metrics` endpoint is intended for the internal observability network and must not be exposed
as a public gateway route.

Recipient addresses must be redacted in logs. Full recipients are allowed only inside the encrypted
outbox payload and the in-memory Nodemailer request required for delivery.

## Architecture Overview

1. **Transactional Outbox Pattern**: Outgoing emails are stored as encrypted payloads in the `mail_outbox` table within the same database transaction as the business entity changes (e.g., creating a password reset token).
2. **Event-driven Relay**: The outbox relay service processes pending records, locks them, and adds jobs to the BullMQ queue (`mail-queue`) when notified of database inserts via PostgreSQL `LISTEN/NOTIFY`.
3. **Queue Worker / Delivery Engine**: The BullMQ queue processor pulls jobs from the queue, decrypts the payloads, and uses Nodemailer to deliver emails to the configured SMTP provider (with automatic retry backoff, jitter, and circuit breaker protection).
4. **Secret Protection**: All sensitive templates are processed such that secret values (e.g., tokens, OTP codes) are only embedded in the email body dynamically and encrypted within the database outbox payload column. No raw secrets are stored in plaintext databases or logs.

---

## Extension Points

The module is designed to support future email-sending flows (such as OTP, login approvals, or security notifications) without changes to the underlying delivery infrastructure or concrete SMTP provider bindings.

### 1. Typed Mail Templates

All templates must be created using React Email and located under the `templates/` directory.

To add a new email template:
1. Create a React component template under `templates/my-new-flow.template.tsx`.
2. Define a typescript props interface for all variables.
3. Ensure the template wraps its content in `MailLayoutTemplate` for consistent design aesthetics.

Example template:
```tsx
import { Text } from '@react-email/components';
import * as React from 'react';
import { MailLayoutTemplate } from './mail-layout.template';

interface MyNewFlowTemplateProps {
    userName: string;
    actionLink: string;
}

export const MyNewFlowTemplate = ({ userName, actionLink }: MyNewFlowTemplateProps) => (
    <MailLayoutTemplate previewText="Action Required" heading="Action Needed">
        <Text>Hi {userName},</Text>
        <Text>Please click the link below to complete your action:</Text>
        <Text><a href={actionLink}>{actionLink}</a></Text>
    </MailLayoutTemplate>
);
```

### 2. Typed Service Methods

Methods for compiling and sending emails should be declared inside `MailTemplateService`. Do not invoke Nodemailer or `MailService` directly from domain modules.

To expose a new email flow:
1. Add a typed method to `MailTemplateService` under `services/mail-template.service.ts`.
2. Accept a database `transaction` argument of type `RepositoryTransaction` to support the transactional outbox pattern.
3. Create the React element, render it to HTML and plain text, and call `this.mailService.createDeliveryRequest()`.

Example service method:
```typescript
async sendMyNewFlowEmail(
    to: string,
    userName: string,
    actionLink: string,
    transaction?: RepositoryTransaction,
) {
    const component = React.createElement(MyNewFlowTemplate, {
        userName,
        actionLink,
    });

    const html = await this.mailRenderService.renderHtml(component);
    const text = await this.mailRenderService.renderText(component);

    return await this.mailService.createDeliveryRequest(
        {
            to,
            subject: 'Action Required',
            html,
            text,
        },
        transaction,
    );
}
```

### 3. Reusing MAIL_SERVICE and Outbox Transactions for Future Flows

Future security/auth flows (such as login approvals, OTP verification, or risk alerts) must not send emails synchronously or bypass the transactional outbox.

Always structure the request handler as follows:
```typescript
await this.db.transaction(async (tx) => {
    // 1. Assert email system readiness
    await this.mailReadinessService.assertMailReadyForCriticalFlow();

    // 2. Perform business logic and persist challenge state
    const challenge = await this.challengeRepository.create({ userId, ... }, tx);

    // 3. Queue email outbox in the same transaction
    await this.mailTemplateService.sendMyNewFlowEmail(
        user.email,
        user.name,
        challenge.link,
        tx
    );
});
```

This guarantees:
- **Consistency**: If the email outbox write fails, the challenge token is rolled back, preventing orphaned/unreachable challenges.
- **Enumeration Safety**: Read-readiness checks are done upfront before looking up emails.
- **Provider Decoupling**: Application flows are not coupled to Nodemailer or any specific SMTP provider; they interact solely with `MailTemplateService`.
