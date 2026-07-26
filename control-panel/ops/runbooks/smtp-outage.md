# SMTP Outage Runbook

## Symptoms

- `smtp_delivery_success_rate` drops.
- `smtp_delivery_errors_total` increases.
- Mail readiness reports `unhealthy`.
- Password reset, email change, and verification mails are delayed.

## Immediate Checks

```bash
curl -sS http://localhost:3000/metrics
docker logs pampilo-platform-api-1 --tail 300 | rg 'SmtpMailService|MailProcessorService|MailOutboxRelayService'
```

## Recovery

1. Check current admin mail settings in the admin UI.
2. Verify SMTP credentials and TLS settings.
3. Send a test email from `/admin/settings/mail`.
4. Watch `smtp_delivery_success_rate`, `smtp_delivery_errors_total`, and outbox lag.
5. If credentials were exposed or rotated, update settings and verify no plaintext secrets appear in logs.

## Customer Impact

Authentication can continue, but critical flows that require email challenge should degrade to manual recovery or return `503` depending on policy.
