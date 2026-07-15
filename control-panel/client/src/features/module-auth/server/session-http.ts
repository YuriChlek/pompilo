import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';
import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';
import {
    buildRefreshCookieHeader,
    getSetCookieHeaders,
} from '@/features/module-auth/server/auth-cookie-header';

export function hasCookie(request: NextRequest, cookieName: COOKIE_NAMES): boolean {
    return Boolean(request.cookies.get(cookieName)?.value);
}

export function buildRefreshHeaders(request: NextRequest, scope: AuthScope): HeadersInit {
    const refreshCookie = buildRefreshCookieHeader(request.cookies, scope);
    const deviceIdCookie = request.cookies.get(COOKIE_NAMES.DEVICE_ID)?.value;

    let cookieHeader = refreshCookie;
    if (deviceIdCookie) {
        cookieHeader = cookieHeader
            ? `${cookieHeader}; ${COOKIE_NAMES.DEVICE_ID}=${deviceIdCookie}`
            : `${COOKIE_NAMES.DEVICE_ID}=${deviceIdCookie}`;
    }

    const userAgent = request.headers?.get ? (request.headers.get('user-agent') ?? '') : '';
    const xForwardedFor = request.headers?.get ? (request.headers.get('x-forwarded-for') ?? '') : '';
    const xRealIp = request.headers?.get ? (request.headers.get('x-real-ip') ?? '') : '';

    return {
        cookie: cookieHeader,
        'user-agent': userAgent,
        'x-forwarded-for': xForwardedFor,
        'x-real-ip': xRealIp,
    };
}

export function appendSetCookies(response: NextResponse, setCookies: string[]): void {
    for (const cookie of setCookies) {
        response.headers.append('set-cookie', cookie);
    }
}

export { getSetCookieHeaders };
