import { getRefreshPathByRequestPath } from '@/features/module-auth/lib/auth-refresh';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';

function getRequestUrl(input: RequestInfo | URL): URL {
    const fallbackOrigin =
        typeof window === 'undefined' ? 'http://localhost' : window.location.origin;

    if (input instanceof Request) {
        return new URL(input.url, fallbackOrigin);
    }

    return new URL(input, fallbackOrigin);
}

function getRefreshUrl(input: RequestInfo | URL): URL {
    const requestUrl = getRequestUrl(input);
    const requestPath = requestUrl.pathname.startsWith('/api')
        ? requestUrl.pathname.slice('/api'.length) || '/'
        : requestUrl.pathname;
    const refreshPath = getRefreshPathByRequestPath(requestPath);

    return new URL(`/api${refreshPath}`, requestUrl.origin);
}

export async function fetchWithAuthRefresh(
    input: RequestInfo | URL,
    init?: RequestInit,
): Promise<Response> {
    let response = await fetch(input, init);

    if (response.status !== 401 && response.status !== 403) {
        return response;
    }

    const requestUrl = getRequestUrl(input);
    if (requestUrl.pathname.includes('/refresh')) {
        return response;
    }

    const refreshResponse = await fetch(getRefreshUrl(input), {
        method: 'POST',
        credentials: 'include',
    });

    if (!refreshResponse.ok) {
        if (refreshResponse.status === 401 && typeof document !== 'undefined') {
            document.cookie = `${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
            document.cookie = `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
            document.cookie = `${COOKIE_NAMES.ADMIN_ACCESS_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
            document.cookie = `${COOKIE_NAMES.ADMIN_REFRESH_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;

            const currentPath = window.location.pathname;
            const loginPath = currentPath.startsWith('/admin') ? '/admin/login' : '/login';
            window.location.href = loginPath;
        }
        return response;
    }

    response = await fetch(input, init);

    return response;
}
