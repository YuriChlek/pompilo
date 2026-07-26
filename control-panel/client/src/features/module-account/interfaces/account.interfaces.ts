export interface UserSession {
    id: string;
    ipAddress: string | null;
    userAgent: string | null;
    createdAt: string;
    currentSession: boolean;
    lastSeenAt?: string;
    trustedAt?: string | null;
    trustExpiresAt?: string | null;
    riskScore?: number;
    approximateLocation?: string | null;
}
