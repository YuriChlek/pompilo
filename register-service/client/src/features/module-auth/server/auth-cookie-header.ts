import { AUTH_SCOPE_CONFIG } from '@/features/module-auth/config/auth-scope.config';
import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';

type CookieRecord = {
    name: string;
    value: string;
};

type CookieSource = {
    getAll(): CookieRecord[];
    get(name: string): { value: string } | undefined;
};

const API_PROXY_PASSTHROUGH_COOKIES = new Set(['theme']);

const ALLOWED_DEVICE_ID_PATHS = new Set([
    'login',
    'admin/login',
    'register',
    'auth/register',
    'refresh',
    'admin/refresh',
    'checkpoint/verify',
    'admin/checkpoint/verify',
    'checkpoint/resend',
    'admin/checkpoint/resend',
]);

function parseCookieHeader(cookieHeader: string | null): Map<string, string> {
    const cookies = new Map<string, string>();

    if (!cookieHeader) {
        return cookies;
    }

    for (const item of cookieHeader.split(';')) {
        const [rawName, ...rawValueParts] = item.trim().split('=');

        if (!rawName) {
            continue;
        }

        cookies.set(rawName, rawValueParts.join('='));
    }

    return cookies;
}

function serializeCookies(cookies: CookieRecord[]): string {
    return cookies.map(({ name, value }) => `${name}=${value}`).join('; ');
}

function serializeCookieMap(cookies: Map<string, string>): string {
    return serializeCookies(
        [...cookies.entries()].map(([name, value]) => ({
            name,
            value,
        })),
    );
}

export function buildAccessCookieHeader(cookieSource: CookieSource, scope: AuthScope): string {
    const allCookies = typeof cookieSource.getAll === 'function' ? cookieSource.getAll() : [];
    const accessCookieName = AUTH_SCOPE_CONFIG[scope].accessCookie;
    return serializeCookies((allCookies || []).filter(({ name }) => name === accessCookieName));
}

export function buildRefreshCookieHeader(cookieSource: CookieSource, scope: AuthScope): string {
    const refreshCookieName = AUTH_SCOPE_CONFIG[scope].refreshCookie;
    const refreshCookie = cookieSource.get(refreshCookieName);

    return refreshCookie ? `${refreshCookieName}=${refreshCookie.value}` : '';
}

export function filterApiProxyCookieHeader(
    cookieHeader: string | null,
    pathname: string,
): string | null {
    const cookies = parseCookieHeader(cookieHeader);

    if (cookies.size === 0) {
        return null;
    }

    const isDeviceIdAllowed = ALLOWED_DEVICE_ID_PATHS.has(pathname);

    if (pathname === 'admin/refresh') {
        const refreshValue = cookies.get(AUTH_SCOPE_CONFIG.admin.refreshCookie);
        const result = new Map<string, string>();
        if (refreshValue) {
            result.set(AUTH_SCOPE_CONFIG.admin.refreshCookie, refreshValue);
        }
        if (isDeviceIdAllowed) {
            const deviceIdValue = cookies.get(COOKIE_NAMES.DEVICE_ID);
            if (deviceIdValue) {
                result.set(COOKIE_NAMES.DEVICE_ID, deviceIdValue);
            }
        }
        return result.size > 0 ? serializeCookieMap(result) : null;
    }

    if (pathname === 'refresh') {
        const refreshValue = cookies.get(AUTH_SCOPE_CONFIG.customer.refreshCookie);
        const result = new Map<string, string>();
        if (refreshValue) {
            result.set(AUTH_SCOPE_CONFIG.customer.refreshCookie, refreshValue);
        }
        if (isDeviceIdAllowed) {
            const deviceIdValue = cookies.get(COOKIE_NAMES.DEVICE_ID);
            if (deviceIdValue) {
                result.set(COOKIE_NAMES.DEVICE_ID, deviceIdValue);
            }
        }
        return result.size > 0 ? serializeCookieMap(result) : null;
    }

    const scope: AuthScope = pathname.startsWith('admin/') ? 'admin' : 'customer';
    const allowedAccessCookie = AUTH_SCOPE_CONFIG[scope].accessCookie;

    for (const key of [...cookies.keys()]) {
        if (key === COOKIE_NAMES.DEVICE_ID) {
            if (!isDeviceIdAllowed) {
                cookies.delete(key);
            }
        } else if (key !== allowedAccessCookie && !API_PROXY_PASSTHROUGH_COOKIES.has(key)) {
            cookies.delete(key);
        }
    }

    return cookies.size > 0 ? serializeCookieMap(cookies) : null;
}

export function getSetCookieHeaders(headers: Headers): string[] {
    const getSetCookie = (headers as Headers & { getSetCookie?: () => string[] }).getSetCookie;

    if (typeof getSetCookie === 'function') {
        const setCookies = getSetCookie.call(headers);

        if (Array.isArray(setCookies)) {
            return setCookies;
        }
    }

    const setCookie = headers.get('set-cookie');

    return setCookie ? [setCookie] : [];
}

export function getCookieHeaderFromSetCookieHeaders(
    headers: Headers,
    cookieName: string,
): string | null {
    const setCookie = getSetCookieHeaders(headers).find(cookie =>
        cookie.startsWith(`${cookieName}=`),
    );

    if (!setCookie) {
        return null;
    }

    return setCookie.split(';')[0] ?? null;
}

export function mergeCookieHeaders(
    cookieHeader: string | null | undefined,
    cookieToUpsert: string,
): string {
    const cookies = new Map<string, string>();

    if (cookieHeader) {
        for (const item of cookieHeader.split(';')) {
            const [rawName, ...rawValueParts] = item.trim().split('=');

            if (rawName) {
                cookies.set(rawName, rawValueParts.join('='));
            }
        }
    }

    const [name, ...valueParts] = cookieToUpsert.split('=');

    if (name) {
        cookies.set(name, valueParts.join('='));
    }

    return serializeCookies(
        [...cookies.entries()].map(([cookieName, value]) => ({
            name: cookieName,
            value,
        })),
    );
}
