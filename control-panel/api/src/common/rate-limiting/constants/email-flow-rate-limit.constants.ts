import type {
    EmailFlowRateLimitConfig,
    EmailRateLimitedFlow,
} from '@/common/rate-limiting/interfaces/email-flow-rate-limit.interfaces';

export const EMAIL_FLOW_RATE_LIMIT_METADATA_KEY = Symbol('EMAIL_FLOW_RATE_LIMIT_METADATA_KEY');

export const EMAIL_FLOW_RATE_LIMIT_MESSAGE = 'Too many requests. Please try again later.';

export const EMAIL_FLOW_RATE_LIMIT_CONFIG: Record<EmailRateLimitedFlow, EmailFlowRateLimitConfig> =
    {
        registration: {
            ip: { maxRequests: 20, windowSeconds: 60 * 60 },
            recipient: { maxRequests: 3, windowSeconds: 60 * 60 },
        },
        password_reset: {
            ip: { maxRequests: 10, windowSeconds: 15 * 60 },
            recipient: { maxRequests: 3, windowSeconds: 60 * 60 },
        },
        email_change: {
            ip: { maxRequests: 10, windowSeconds: 15 * 60 },
            recipient: { maxRequests: 3, windowSeconds: 60 * 60 },
        },
        resend_verification: {
            ip: { maxRequests: 5, windowSeconds: 15 * 60 },
            recipient: { maxRequests: 3, windowSeconds: 15 * 60 },
        },
    } as const;
