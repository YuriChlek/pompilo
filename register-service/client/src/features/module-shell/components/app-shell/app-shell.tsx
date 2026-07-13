'use client';

import { useState } from 'react';
import { setCookie } from 'cookies-next';
import { Sidebar } from '@/features/module-shell/components/sidebar/sidebar';
import { MobileNav } from '@/features/module-shell/components/mobile-nav/mobile-nav';
import { MenuTypes } from '@/features/module-menu/enums/menu.enums';
import type { User } from '@/features/module-auth/interfaces/auth.interfaces';
import styles from './styles.module.css';

interface AppShellProps {
    children: React.ReactNode;
    initialCollapsed: boolean;
    user: User | null;
    menuType: MenuTypes;
}

export const AppShell = ({
    children,
    initialCollapsed,
    user,
    menuType,
}: AppShellProps) => {
    const [isCollapsed, setIsCollapsed] = useState(initialCollapsed);

    const handleToggle = () => {
        const newState = !isCollapsed;
        setIsCollapsed(newState);
        setCookie('sidebar-collapsed', newState, { maxAge: 60 * 60 * 24 * 30 });
    };

    return (
        <div className={styles.shell}>
            <Sidebar
                menuType={menuType}
                user={user}
                isCollapsed={isCollapsed}
                onToggle={handleToggle}
            />
            <main className={styles.main}>
                <div className={styles.content}>{children}</div>
                <MobileNav menuType={menuType} />
            </main>
        </div>
    );
};
