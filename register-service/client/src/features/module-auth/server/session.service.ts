import type { NextRequest } from 'next/server';
import { AUTH_SCOPE_CONFIG } from '@/features/module-auth/config/auth-scope.config';
import { getRefreshPathByScope } from '@/features/module-auth/lib/auth-refresh';
import { getScopes } from '@/features/module-auth/lib/auth-scope';
import { apiBaseUrl } from '@/lib/config/api-base-url.config';
import {
    buildRefreshHeaders,
    getSetCookieHeaders,
    hasCookie,
} from '@/features/module-auth/server/session-http';
import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';
import type { RefreshResult } from '@/features/module-auth/types/session.types';
import { isJwtExpired } from '@/features/module-auth/lib/jwt';

export async function resolveSessionState(
    request: NextRequest,
): Promise<Record<AuthScope, RefreshResult>> {
    const results = await Promise.all(
        getScopes().map(async (scope) => {
            const session = await ensureAuthenticatedSession(request, scope);

            return [scope, session] as const;
        }),
    );

    return Object.fromEntries(results) as Record<AuthScope, RefreshResult>;
}

/**
 * Determines the authentication state for a specific scope.
 * 
 * ARCHITECTURAL ROLE:
 * This function acts as a pre-emptive check in the Next.js Middleware (Proxy).
 * It uses JWT expiration (from unverified payload) to decide if a refresh is needed
 * before the request is passed downstream to Server Components or the Backend.
 * 
 * - If the token is expired according to payload -> refresh.
 * - If the token is missing but refresh cookie exists -> refresh.
 * - If both are missing -> unauthenticated.
 * 
 * Final authority on token validity (signature check) remains with the Backend (/me).
 */
export async function ensureAuthenticatedSession(
    request: NextRequest,
    scope: AuthScope,
): Promise<RefreshResult> {
    const authConfig = AUTH_SCOPE_CONFIG[scope];
    const accessToken = request.cookies.get(authConfig.accessCookie)?.value;
    const hasRefreshToken = hasCookie(request, authConfig.refreshCookie);

    if (accessToken) {
        // SCENARIO B: Access token is present. Check if it is expired.
        // We use a small buffer (10s) to avoid race conditions with downstream requests
        if (!isJwtExpired(accessToken, 10)) {
            return {
                authenticated: true,
                setCookies: [],
            };
        }
    }

    // Access token is missing or expired.
    if (!hasRefreshToken) {
        return {
            authenticated: false,
            setCookies: [],
        };
    }

    // We have a refresh token, so attempt to refresh the session.
    return refreshAccessToken(request, scope);
}

export class RedisOutageError extends Error {
    constructor(message = 'Service Unavailable') {
        super(message);
        this.name = 'RedisOutageError';
    }
}

export async function refreshAccessToken(
    request: NextRequest,
    scope: AuthScope,
): Promise<RefreshResult> {
    const authConfig = AUTH_SCOPE_CONFIG[scope];
    let response: Response;

    const refreshHeaders = buildRefreshHeaders(request, scope);
    const refreshUrl = new URL(getRefreshPathByScope(scope), apiBaseUrl);

    try {
        response = await fetch(refreshUrl, {
            method: 'POST',
            headers: refreshHeaders,
            cache: 'no-store',
        });
    } catch (error) {
        throw new RedisOutageError(error instanceof Error ? error.message : 'Network Error');
    }

    if (response.status === 503) {
        throw new RedisOutageError('503 Service Unavailable');
    }

    const setCookies = getSetCookieHeaders(response.headers) || [];
    const payload = (await response.json().catch(() => null)) as {
        success?: boolean;
        data?: boolean;
    } | null;
    const hasAccessCookie = setCookies.some(cookie =>
        cookie.startsWith(`${authConfig.accessCookie}=`),
    );

    return {
        authenticated:
            response.ok &&
            payload?.success === true &&
            payload.data === true &&
            hasAccessCookie,
        setCookies,
    };
}
