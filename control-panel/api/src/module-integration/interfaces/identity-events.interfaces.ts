export type IdentityEventType =
    | 'identity.v1.UserRegistered'
    | 'identity.v1.EmailVerified'
    | 'identity.v1.TradingAccessGranted'
    | 'identity.v1.TradingAccessChanged'
    | 'identity.v1.TradingAccessSuspended'
    | 'identity.v1.UserDisabled'
    | 'identity.v1.UserDeleted';

export interface IdentityEventEnvelope<TPayload extends Record<string, unknown> = Record<string, unknown>> {
    eventId: string;
    eventType: IdentityEventType;
    eventVersion: 1;
    occurredAt: string;
    aggregateId: string;
    tenantId: string;
    payload: TPayload;
}

export interface IdentityContext {
    userId: string;
    tenantId: string;
    membershipRole: 'OWNER' | 'ADMIN' | 'MEMBER';
}

export interface TradingOnboardingTokenPayload {
    iss: 'identity-service';
    aud: 'trading-service';
    scope: 'trading:onboarding';
    sub: string;
    userId: string;
    tenantId: string;
    membershipRole: 'OWNER' | 'ADMIN' | 'MEMBER';
    emailVerified: true;
}
