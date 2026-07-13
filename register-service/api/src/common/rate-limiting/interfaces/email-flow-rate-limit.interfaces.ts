export type EmailRateLimitedFlow =
    | 'registration'
    | 'password_reset'
    | 'email_change'
    | 'resend_verification';

export type EmailRateLimitDimension = 'ip' | 'recipient';

export interface EmailFlowRateLimitOptions {
    flow: EmailRateLimitedFlow;
    recipientBodyField?: 'email' | 'newEmail';
}

export interface EmailFlowRateLimitBucketConfig {
    maxRequests: number;
    windowSeconds: number;
}

export interface EmailFlowRateLimitConfig {
    ip: EmailFlowRateLimitBucketConfig;
    recipient: EmailFlowRateLimitBucketConfig;
}

export interface EmailFlowRateLimitCheckInput {
    flow: EmailRateLimitedFlow;
    ipAddress: string;
    recipientEmail?: string;
}

export interface EmailFlowRateLimitCheckResult {
    limited: boolean;
    retryAfterSeconds: number;
}
