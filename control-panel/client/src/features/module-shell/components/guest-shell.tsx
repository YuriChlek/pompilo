import 'server-only';

import type { ReactNode } from 'react';
import { cookies } from 'next/headers';
import { getActiveCustomerSession } from '@/features/module-auth/server/active-customer-session';
import { AppShell } from '@/features/module-shell/components/app-shell/app-shell';
import { MenuTypes } from '@/features/module-menu/enums/menu.enums';

interface GuestShellProps {
    children: ReactNode;
}

/**
 * Server component that renders the identity shell for guest and customer views.
 */
export async function GuestShell({ children }: GuestShellProps) {
    const [session, cookieStore] = await Promise.all([
        getActiveCustomerSession(),
        cookies(),
    ]);

    const initialCollapsed = cookieStore.get('sidebar-collapsed')?.value === 'true';

    return (
        <AppShell
            initialCollapsed={initialCollapsed}
            user={session.user}
            menuType={session.user ? MenuTypes.CUSTOMER : MenuTypes.GUEST}
        >
            {children}
        </AppShell>
    );
}
