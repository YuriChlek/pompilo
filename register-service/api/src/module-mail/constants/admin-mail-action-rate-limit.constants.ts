export const ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY = Symbol(
    'ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY',
);

export type AdminMailRateLimitedAction = 'test_email';

export interface AdminMailActionRateLimitOptions {
    action: AdminMailRateLimitedAction;
}

export const ADMIN_MAIL_ACTION_RATE_LIMIT_CONFIG: Record<
    AdminMailRateLimitedAction,
    { maxRequests: number; windowSeconds: number }
> = {
    test_email: { maxRequests: 3, windowSeconds: 15 * 60 },
} as const;

export const ADMIN_MAIL_ACTION_RATE_LIMIT_MESSAGE =
    'Too many admin mail requests. Please try again later.';
