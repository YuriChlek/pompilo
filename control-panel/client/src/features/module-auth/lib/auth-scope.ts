import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';

const AUTH_SCOPES: AuthScope[] = ['admin', 'customer'];

export function getScopes(): AuthScope[] {
    return AUTH_SCOPES;
}
