import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';
import type { RouteContext } from '@/features/module-auth/types/route-access.types';

const STATIC_PATH_REGEX =
    /^\/(_next|api|favicon\.ico$|.*\.(js|css|map|png|jpg|jpeg|gif|svg|woff|woff2)$)/;
const ADMIN_PATH_REGEX = /^\/admin(?:\/|$)/;
const ACCOUNT_PATH_REGEX = /^\/account(?:\/|$)/;

export function getRouteContext(pathname: string, search: string): RouteContext {
    return {
        pathname,
        from: `${pathname}${search}`,
    };
}

export function isStaticPath(pathname: string): boolean {
    return STATIC_PATH_REGEX.test(pathname);
}

export function isProtectedRoute(scope: AuthScope, pathname: string): boolean {
    if (scope === 'admin') {
        return ADMIN_PATH_REGEX.test(pathname) && pathname !== '/admin/login';
    }

    return ACCOUNT_PATH_REGEX.test(pathname);
}

export function isGuestRoute(scope: AuthScope, pathname: string): boolean {
    if (scope === 'admin') {
        return pathname === '/admin' || pathname === '/admin/' || pathname === '/admin/login';
    }

    return (
        pathname === '/login' ||
        pathname === '/register' ||
        pathname === '/forgot-password' ||
        pathname === '/reset-password' ||
        pathname === '/auth/forgot-password' ||
        pathname === '/auth/reset-password'
    );
}
