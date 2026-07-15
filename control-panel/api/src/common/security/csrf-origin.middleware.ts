import type { NextFunction, Request, Response } from 'express';
import { COOKIE_NAMES } from '@/module-auth/enums/auth.enums';

const UNSAFE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);
const AUTH_COOKIE_NAMES = new Set<string>([
    COOKIE_NAMES.ADMIN_ACCESS_TOKEN,
    COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
    COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN,
    COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
]);

export function createCsrfOriginMiddleware(options: {
    enabled: boolean;
    allowedOrigins: string[];
}) {
    const allowedOrigins = new Set(
        options.allowedOrigins.map(origin => normalizeOrigin(origin)).filter(Boolean),
    );

    return (request: Request, response: Response, next: NextFunction): void => {
        if (
            !options.enabled ||
            !UNSAFE_METHODS.has(request.method.toUpperCase()) ||
            !hasAuthCookie(request)
        ) {
            next();
            return;
        }

        const requestOrigin = getRequestOrigin(request);

        if (requestOrigin && allowedOrigins.has(requestOrigin)) {
            next();
            return;
        }

        response.status(403).json({
            success: false,
            statusCode: 403,
            message: 'CSRF origin check failed',
        });
    };
}

function hasAuthCookie(request: Request): boolean {
    const cookies = request.cookies as Record<string, unknown> | undefined;

    if (!cookies) {
        return false;
    }

    return Object.keys(cookies).some(cookieName => AUTH_COOKIE_NAMES.has(cookieName));
}

function getRequestOrigin(request: Request): string | null {
    const origin = normalizeOrigin(request.headers.origin);

    if (origin) {
        return origin;
    }

    return normalizeOrigin(request.headers.referer);
}

function normalizeOrigin(value: string | string[] | undefined): string | null {
    const rawValue = Array.isArray(value) ? value[0] : value;

    if (!rawValue) {
        return null;
    }

    try {
        const url = new URL(rawValue);
        return url.origin;
    } catch {
        return null;
    }
}
