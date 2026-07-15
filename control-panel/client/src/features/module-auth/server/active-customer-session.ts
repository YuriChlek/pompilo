import 'server-only';

import { cache } from 'react';
import { cookies } from 'next/headers';
import { unstable_rethrow } from 'next/navigation';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import { CustomerSessionState } from '@/features/module-auth/enums/customer-session-state.enums';
import { resolveCustomerSessionState } from '@/features/module-auth/server/customer-session-resolver';
import { getCurrentUser } from '@/features/module-auth/api-service/server';
import type { User } from '@/features/module-auth/interfaces/auth.interfaces';

export interface CustomerSession {
    user: User | null;
    role: UserRoles.USER | null;
}

/**
 * Request-scoped cached server helper to resolve the active customer session.
 *
 * It ensures that:
 * - Only customer sessions are resolved (ignores guest/admin).
 * - Redundant /me API calls are prevented within the same server render.
 */
export const getActiveCustomerSession = cache(async (): Promise<CustomerSession> => {
    const cookieStore = await cookies();
    const state = resolveCustomerSessionState(cookieStore);

    if (state === CustomerSessionState.NONE) {
        return { user: null, role: null };
    }

    try {
        const user = await getCurrentUser();

        if (!user || user.role !== UserRoles.USER) {
            return { user: null, role: null };
        }

        return { user, role: user.role };
    } catch (error) {
        unstable_rethrow(error);
        
        // If it was not a Next.js internal error (redirect/notFound), 
        // we should rethrow technical errors to trigger Error Boundaries.
        // getCurrentUser already masks 401/403 as null, so if we are here, it's a real error.
        throw error;
    }
});
