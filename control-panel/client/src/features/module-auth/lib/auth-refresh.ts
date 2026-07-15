import { AUTH_SCOPE_CONFIG } from '@/features/module-auth/config/auth-scope.config';
import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';

export function getAuthScopeFromPath(path: string): AuthScope {
    if (path.includes('/admin')) {
        return 'admin';
    }

    return 'customer';
}

export function getRefreshPathByScope(scope: AuthScope): string {
    return AUTH_SCOPE_CONFIG[scope].refreshPath;
}

export function getRefreshPathByRequestPath(path: string): string {
    const scope = getAuthScopeFromPath(path);
    return getRefreshPathByScope(scope);
}
