import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';

export type SessionMetadata = {
    ipAddress?: string | null;
    userAgent?: string | null;
    lastCountry?: string | null;
    lastRegion?: string | null;
    lastCity?: string | null;
    riskScore?: number;
    riskReason?: string | null;
};

export type RevokeCurrentSessionInput = {
    sessionId: string;
    userId: string;
    realm: 'customer' | 'admin';
    accessTokenJti?: string;
    ipAddress?: string;
    userAgent?: string;
};

export type RevokeSpecificSessionInput = {
    sessionId: string;
    userId: string;
    realm: 'customer' | 'admin';
};

export type RevokeOtherSessionsInput = {
    userId: string;
    realm: 'customer' | 'admin';
    currentSessionId: string;
    reauthConfirmationToken?: string;
};

export type RevokeUserSessionsInput = {
    userId: string;
    eventType?: SecurityEventType;
    fallbackRealm?: 'customer' | 'admin';
};

export type RevokeAllSessionsInput = {
    userId: string;
    realm: 'customer' | 'admin';
    currentSessionId: string;
};
