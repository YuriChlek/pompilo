import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';
import type { RouteContext } from '@/features/module-auth/types/route-access.types';
import type { RefreshResult } from '@/features/module-auth/types/session.types';
import { CUSTOMER_DEFAULT_MENU_ITEM } from '@/features/module-menu/config/menu.config';

export function resolveProxyRedirect(
    request: NextRequest,
    route: RouteContext,
    sessionState: Record<AuthScope, RefreshResult>,
): NextResponse | null {
    const isAccountRoute = /^\/account(?:\/|$)/.test(route.pathname);
    const isCustomerProtectedRoute = isAccountRoute;

    // 1. Admin checks
    const isAdminRoute = /^\/admin(?:\/|$)/.test(route.pathname);
    const isAdminLoginRoute = route.pathname === '/admin/login';

    if (isAdminRoute && !isAdminLoginRoute) {
        if (!sessionState.admin.authenticated) {
            const loginUrl = new URL('/admin/login', request.url);
            loginUrl.searchParams.set('from', route.from);
            return NextResponse.redirect(loginUrl);
        }
        return null;
    }

    if (isAdminLoginRoute && sessionState.admin.authenticated) {
        return NextResponse.redirect(new URL('/admin/dashboard', request.url));
    }

    // 2. Customer guest routes (/login, /register)
    const isCustomerGuestRoute =
        route.pathname === '/login' ||
        route.pathname === '/register' ||
        route.pathname === '/forgot-password' ||
        route.pathname === '/reset-password' ||
        route.pathname === '/auth/forgot-password' ||
        route.pathname === '/auth/reset-password';
    if (isCustomerGuestRoute && sessionState.customer.authenticated) {
        return NextResponse.redirect(new URL(CUSTOMER_DEFAULT_MENU_ITEM.href, request.url));
    }

    // 3. Customer protected routes
    if (isCustomerProtectedRoute) {
        if (!sessionState.customer.authenticated) {
            const loginUrl = new URL('/login', request.url);
            loginUrl.searchParams.set('from', route.from);
            return NextResponse.redirect(loginUrl);
        }

    }

    return null;
}
