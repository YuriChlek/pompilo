import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleAuthProxy } from '@/features/module-auth/server/handle-auth-proxy';
import { resolveSessionState } from '@/features/module-auth/server/session.service';
import { resolveProxyRedirect } from '@/features/module-auth/server/proxy-redirects';
import { NextRequest, NextResponse } from 'next/server';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';

vi.mock('@/features/module-auth/server/session.service', () => {
    class MockRedisOutageError extends Error {
        constructor(message = 'Service Unavailable') {
            super(message);
            this.name = 'RedisOutageError';
        }
    }
    return {
        resolveSessionState: vi.fn(),
        RedisOutageError: MockRedisOutageError,
    };
});

vi.mock('@/features/module-auth/server/proxy-redirects', () => ({
    resolveProxyRedirect: vi.fn(),
}));

describe('handleAuthProxy (Phase 4 fix)', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('Scenario A: root page refresh should override downstream request headers', async () => {
        const mockSessionState = {
            customer: {
                authenticated: true,
                setCookies: [`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token; Path=/`],
            },
            admin: { authenticated: false, setCookies: [] },
        };

        vi.mocked(resolveSessionState).mockResolvedValue(mockSessionState);
        vi.mocked(resolveProxyRedirect).mockReturnValue(null); // No redirect for root page

        const request = new NextRequest(new URL('https://localhost/'), {
            headers: {
                cookie: `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=valid-refresh`,
            },
        });

        const response = await handleAuthProxy(request);

        // 1. Response should be 'next' (implicit, as it's not a redirect)
        expect(response.headers.get('location')).toBeNull();

        // 2. Browser should receive Set-Cookie
        expect(response.headers.get('set-cookie')).toContain(`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token; Path=/`);

        /**
         * 3. Downstream request headers MUST be overridden.
         * In Next.js middleware, this is achieved by returning NextResponse.next({ request: { headers } }).
         * We verify this by checking the special internal header Next.js uses for this override.
         */
        expect(response.headers.get('x-middleware-override-headers')).toContain('cookie');
        expect(response.headers.get('x-middleware-request-cookie')).toContain(`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token`);
        expect(response.headers.get('x-middleware-request-cookie')).toContain(`${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=valid-refresh`);
    });

    it('Scenario B: expired token refresh followed by redirect should preserve cookies', async () => {
        const expiredToken = btoa(JSON.stringify({ alg: 'HS256' })) + '.' + 
            btoa(JSON.stringify({ exp: Math.floor(Date.now() / 1000) - 10, role: 'user' })) + '.sig';

        const mockSessionState = {
            customer: {
                authenticated: true,
                setCookies: [`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token; Path=/`],
            },
            admin: { authenticated: false, setCookies: [] },
        };

        vi.mocked(resolveSessionState).mockResolvedValue(mockSessionState);
        
        // Simulate redirecting from login back to home because we just refreshed/authenticated
        const redirectUrl = new URL('https://localhost/');
        vi.mocked(resolveProxyRedirect).mockReturnValue(NextResponse.redirect(redirectUrl));

        const request = new NextRequest(new URL('https://localhost/login'), {
            headers: {
                cookie: `${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=${expiredToken}; ${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=valid-refresh`,
            },
        });

        const response = await handleAuthProxy(request);

        expect(response.status).toBe(307);
        expect(response.headers.get('location')).toBe('https://localhost/');
        expect(response.headers.get('set-cookie')).toContain(`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token; Path=/`);
    });

    it('Scenario B: root page refresh with expired token should override downstream request headers', async () => {
        const expiredToken = btoa(JSON.stringify({ alg: 'HS256' })) + '.' + 
            btoa(JSON.stringify({ exp: Math.floor(Date.now() / 1000) - 10, role: 'user' })) + '.sig';

        const mockSessionState = {
            customer: {
                authenticated: true,
                setCookies: [`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token; Path=/`],
            },
            admin: { authenticated: false, setCookies: [] },
        };

        vi.mocked(resolveSessionState).mockResolvedValue(mockSessionState);
        vi.mocked(resolveProxyRedirect).mockReturnValue(null);

        const request = new NextRequest(new URL('https://localhost/'), {
            headers: {
                cookie: `${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=${expiredToken}; ${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=valid-refresh`,
            },
        });

        const response = await handleAuthProxy(request);

        expect(response.headers.get('location')).toBeNull();
        expect(response.headers.get('set-cookie')).toContain(`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token; Path=/`);

        // Verify downstream override
        expect(response.headers.get('x-middleware-override-headers')).toContain('cookie');
        expect(response.headers.get('x-middleware-request-cookie')).toContain(`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-token`);
        // The old expired token should be replaced in the downstream cookie header
        expect(response.headers.get('x-middleware-request-cookie')).not.toContain(expiredToken);
    });

    it('should propagate deviceId, access token, and refresh token to browser Set-Cookie when pre-refresh succeeds', async () => {
        const mockSessionState = {
            customer: {
                authenticated: true,
                setCookies: [
                    `${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-access; Path=/`,
                    `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=new-refresh; Path=/`,
                    `${COOKIE_NAMES.DEVICE_ID}=new-device; Path=/`,
                ],
            },
            admin: { authenticated: false, setCookies: [] },
        };

        vi.mocked(resolveSessionState).mockResolvedValue(mockSessionState);
        vi.mocked(resolveProxyRedirect).mockReturnValue(null);

        const request = new NextRequest(new URL('https://localhost/'), {
            headers: {
                cookie: `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=valid-refresh`,
            },
        });

        const response = await handleAuthProxy(request);

        // All cookies should be propagated to browser Set-Cookie
        const setCookies = response.headers.getSetCookie();
        expect(setCookies).toHaveLength(3);
        expect(setCookies).toContain(`${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=new-access; Path=/`);
        expect(setCookies).toContain(`${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=new-refresh; Path=/`);
        expect(setCookies).toContain(`${COOKIE_NAMES.DEVICE_ID}=new-device; Path=/`);
    });

    it('should return 503 response if resolveSessionState throws RedisOutageError (Phase 27.2)', async () => {
        const { RedisOutageError } = await import('@/features/module-auth/server/session.service');
        vi.mocked(resolveSessionState).mockRejectedValue(new RedisOutageError('Redis Outage'));

        const request = new NextRequest(new URL('https://localhost/'), {
            headers: {
                cookie: `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=valid-refresh`,
            },
        });

        const response = await handleAuthProxy(request);

        expect(response.status).toBe(503);
        const text = await response.text();
        expect(text).toBe('Service Unavailable');
    });

    it('should propagate non-RedisOutageError exceptions', async () => {
        vi.mocked(resolveSessionState).mockRejectedValue(new Error('Unknown Error'));

        const request = new NextRequest(new URL('https://localhost/'), {
            headers: {
                cookie: `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=valid-refresh`,
            },
        });

        await expect(handleAuthProxy(request)).rejects.toThrow('Unknown Error');
    });
});
