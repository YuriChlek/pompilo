import type { ReactNode } from 'react';
import { GuestShell } from '@/features/module-shell/components/guest-shell';

interface CustomerLayoutProps {
    children: ReactNode;
}

export default function CustomerLayout({ children }: CustomerLayoutProps) {
    return (
        <GuestShell>
            {children}
        </GuestShell>
    );
}
