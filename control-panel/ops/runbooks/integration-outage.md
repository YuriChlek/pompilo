# Integration Outage Runbook

## Symptoms

- Trading onboarding webhooks fail.
- Identity outbox events stop being consumed.
- Service-to-service calls return `401`, `403`, or `503`.

## Immediate Checks

```bash
docker logs pampilo-platform-api-1 --tail 300 | rg 'IdentityIntegration|TradingOnboarding|service-token|identity-outbox'
curl -k -sS https://localhost/health/identity
curl -k -sS https://localhost/health/trading
```

## Recovery

1. Confirm `SERVICE_TO_SERVICE_SECRET` and trading onboarding token secret are configured consistently.
2. Check API and trading bot health through gateway routes.
3. Inspect identity outbox records and retry failed consumers.
4. Confirm no duplicate onboarding events were produced before replaying.

## Replay Safety

Replay only idempotent events. Use event ids/idempotency keys from the identity outbox when coordinating with trading services.
