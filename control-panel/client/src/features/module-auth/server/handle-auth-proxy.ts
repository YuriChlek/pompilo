import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getRouteContext, isStaticPath } from '@/features/module-auth/lib/route-access';
import { resolveSessionState, RedisOutageError } from '@/features/module-auth/server/session.service';
import { resolveProxyRedirect } from '@/features/module-auth/server/proxy-redirects';
import { appendSetCookies } from '@/features/module-auth/server/session-http';
import { mergeSetCookiesIntoCookieHeader } from '@/features/module-auth/lib/cookie-merge';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';

export async function handleAuthProxy(request: NextRequest): Promise<NextResponse> {
    const route = getRouteContext(request.nextUrl.pathname, request.nextUrl.search);

    if (isStaticPath(route.pathname)) {
        return NextResponse.next();
    }

    let sessionState;
    try {
        sessionState = await resolveSessionState(request);
    } catch (error) {
        if (error instanceof RedisOutageError) {
            return new NextResponse('Service Unavailable', { status: 503 });
        }
        throw error;
    }
    const allSetCookies = Object.values(sessionState).flatMap(s => s.setCookies);

    // Determine the base response: either a redirect or a simple "next"
    const redirectResponse = resolveProxyRedirect(request, route, sessionState);

    let response: NextResponse;

    if (redirectResponse) {
        response = redirectResponse;
        if (request.cookies.has(COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN) && !sessionState.customer.authenticated) {
            response.cookies.delete(COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN);
            response.cookies.delete(COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN);
        }
        if (request.cookies.has(COOKIE_NAMES.ADMIN_REFRESH_TOKEN) && !sessionState.admin.authenticated) {
            response.cookies.delete(COOKIE_NAMES.ADMIN_ACCESS_TOKEN);
            response.cookies.delete(COOKIE_NAMES.ADMIN_REFRESH_TOKEN);
        }
    } else if (allSetCookies.length > 0) {
        // SCENARIO A: No redirect, but we have new cookies (e.g., refresh on root page).
        // We must override downstream request headers so Server Components see the new cookies immediately.
        const currentCookieHeader = request.headers.get('cookie');
        const updatedCookieHeader = mergeSetCookiesIntoCookieHeader(
            currentCookieHeader,
            allSetCookies,
        );

        const requestHeaders = new Headers(request.headers);
        requestHeaders.set('cookie', updatedCookieHeader);

        response = NextResponse.next({
            request: {
                headers: requestHeaders,
            },
        });
    } else {
        response = NextResponse.next();
    }

    // Always ensure new cookies are sent to the browser
    if (allSetCookies.length > 0) {
        appendSetCookies(response, allSetCookies);
    }

    return response;
}
