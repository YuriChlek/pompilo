import 'server-only';

import { unstable_rethrow } from 'next/navigation';
import { getApiServerClient } from '@/lib/http-client/http-client.server';
import type { User } from '@/features/module-auth/interfaces/auth.interfaces';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { AuthScope } from '@/features/module-auth/types/auth-scope.types';
import type { HttpError } from '@/lib/http-client/interfaces/http-client.interfaces';

function getScopeFromRole(role?: UserRoles): AuthScope | undefined {
    if (role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN) {
        return 'admin';
    }

    if (role === UserRoles.USER) {
        return 'customer';
    }

    return undefined;
}

function getCurrentUserPath(role?: UserRoles): string {
    if (role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN) {
        return '/admin/me';
    }

    return '/me';
}

function isHttpError(error: unknown): error is HttpError {
    return (
        typeof error === 'object' &&
        error !== null &&
        'statusCode' in error &&
        'success' in error &&
        (error as HttpError).success === false
    );
}

export async function getCurrentUser(role?: UserRoles): Promise<User | null> {
    try {
        const apiClient = await getApiServerClient();
        const response = await apiClient.post<User, Record<string, never>>(
            getCurrentUserPath(role),
            {},
            {
                authScope: getScopeFromRole(role),
                cache: 'no-store',
                signal: AbortSignal.timeout(5000),
            },
        );

        return response.data ?? null;
    } catch (error) {
        unstable_rethrow(error);

        if (isHttpError(error)) {
            // 401 Unauthorized (after failed refresh) or 403 Forbidden (role mismatch)
            // should be treated as guest/unauthenticated in the public context.
            if (error.statusCode === 401 || error.statusCode === 403) {
                return null;
            }

            // For other technical errors (5xx, etc.), we don't want to mask them as guests.
            // They should ideally trigger an Error Boundary via rethrow.
        }

        throw error;
    }
}
