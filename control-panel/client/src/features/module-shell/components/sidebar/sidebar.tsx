'use client';

import clsx from 'clsx';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faChevronLeft, faChevronRight } from '@fortawesome/free-solid-svg-icons';
import { Logo } from '@/components/logo/logo';
import { ThemeButton } from '@/components/theme-button/theme-button';
import { LogoutButton } from '@/features/module-auth/components/logout-button/logout-button';
import { getMenu } from '@/features/module-menu/config/menu.config';
import { MenuTypes } from '@/features/module-menu/enums/menu.enums';
import styles from './styles.module.css';
import type { MenuItem } from '@/features/module-menu/interfaces/menu.interfaces';
import type { User } from '@/features/module-auth/interfaces/auth.interfaces';

interface SidebarProps {
    menuType: MenuTypes;
    user: User | null;
    isCollapsed: boolean;
    onToggle: () => void;
}

export const Sidebar = ({ menuType, user, isCollapsed, onToggle }: SidebarProps) => {
    const pathname = usePathname();
    const menuItems = getMenu(menuType);

    const isItemActive = (item: MenuItem) => {
        if (!pathname) return false;
        if (item.href === '/') {
            return pathname === '/';
        }
        return (
            pathname === item.href ||
            pathname.startsWith(`${item.href}/`) ||
            item.matchPaths?.some(p => pathname === p || pathname.startsWith(`${p}/`)) === true
        );
    };

    return (
        <aside className={clsx(styles.sidebar, { [styles.collapsed]: isCollapsed })}>
            <div className={styles.header}>
                <div className={styles.logoWrapper}>
                    <Logo />
                </div>
                <button
                    className={styles.toggleButton}
                    onClick={onToggle}
                    aria-label={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
                >
                    <FontAwesomeIcon icon={isCollapsed ? faChevronRight : faChevronLeft} />
                </button>
            </div>

            <nav className={styles.nav}>
                {menuItems.map(item => {
                    const isActive = isItemActive(item);

                    if (item.disabled) {
                        return (
                            <div
                                key={item.title}
                                className={clsx(styles.navItem, styles.disabled)}
                                title="Coming soon"
                            >
                                <div className={styles.icon}>
                                    {item.icon && typeof item.icon !== 'string' && (
                                        <FontAwesomeIcon icon={item.icon} />
                                    )}
                                </div>
                                <span className={styles.label}>{item.title}</span>
                            </div>
                        );
                    }

                    return (
                        <Link
                            key={item.title}
                            href={item.href}
                            className={clsx(styles.navItem, { [styles.navItemActive]: isActive })}
                        >
                            <div className={styles.iconWrapper}>
                                <div className={styles.icon}>
                                    {item.icon && typeof item.icon !== 'string' && (
                                        <FontAwesomeIcon icon={item.icon} />
                                    )}
                                </div>
                            </div>
                            <span className={styles.label}>{item.title}</span>
                        </Link>
                    );
                })}
            </nav>

            <div className={styles.footer}>
                <div className={styles.footerActions}>
                    {!isCollapsed && <ThemeButton />}
                    <LogoutButton
                        isAuthenticated={Boolean(user)}
                        role={user?.role}
                        forceCompact={isCollapsed}
                    />
                </div>
            </div>
        </aside>
    );
};
