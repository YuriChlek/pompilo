'use client';

import { useState, type MouseEvent } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSun } from '@fortawesome/free-regular-svg-icons';
import { faRobot, faXmark } from '@fortawesome/free-solid-svg-icons';
import styles from './admin-sidebar.module.css';

interface SubMenuItem {
    title: string;
    href: string;
}

interface MenuItem {
    id: string;
    title: string;
    icon: typeof faSun;
    children: SubMenuItem[];
}

const MENU_ITEMS: MenuItem[] = [
    {
        id: 'bots',
        title: 'Bots',
        icon: faRobot,
        children: [
            { title: 'Configuration', href: '/admin/bots/config' },
        ],
    },
    {
        id: 'settings',
        title: 'Settings',
        icon: faSun,
        children: [
            { title: 'Email', href: '/admin/settings/mail' },
            { title: 'Payment methods', href: '/admin/settings/payment' },
        ],
    },
];

export const AdminSidebar = () => {
    const pathname = usePathname();
    const [openSubMenu, setOpenSubMenu] = useState<string | null>(null);

    const isItemActive = (href: string) => {
        if (!pathname) return false;
        return pathname === href || pathname.startsWith(`${href}/`);
    };

    const handleMainMenuClick = (itemId: string, event: MouseEvent) => {
        event.preventDefault();
        setOpenSubMenu(prev => (prev === itemId ? null : itemId));
    };

    const openMenuItem = MENU_ITEMS.find(item => item.id === openSubMenu);

    return (
        <aside className={styles.sidebar}>
            <div className={styles.mainColumn}>
                <div className={styles.header}>
                    <div className={styles.logoWrapper}>
                        <Link
                            href="/admin/dashboard"
                            className={styles.logoBadge}
                            aria-label="Go to admin dashboard"
                        >
                            P
                        </Link>
                    </div>
                </div>

                <nav className={styles.nav} aria-label="Admin navigation">
                    {MENU_ITEMS.map(item => {
                        const isSubmenuOpen = openSubMenu === item.id;
                        const isAnyChildActive = item.children.some(child => isItemActive(child.href));

                        return (
                            <button
                                key={item.id}
                                type="button"
                                onClick={event => handleMainMenuClick(item.id, event)}
                                className={clsx(styles.navItem, {
                                    [styles.navItemActive]: isAnyChildActive,
                                    [styles.navItemOpen]: isSubmenuOpen,
                                })}
                            >
                                <div className={styles.iconWrapper}>
                                    <FontAwesomeIcon icon={item.icon} className={styles.icon} />
                                </div>
                                <span className={styles.label}>{item.title}</span>
                            </button>
                        );
                    })}
                </nav>
            </div>

            {openMenuItem && (
                <div className={styles.submenuColumn}>
                    <div className={styles.submenuHeader}>
                        <span className={styles.submenuTitle}>{openMenuItem.title}</span>
                        <button
                            type="button"
                            onClick={() => setOpenSubMenu(null)}
                            className={styles.closeButton}
                            aria-label="Close menu"
                        >
                            <FontAwesomeIcon icon={faXmark} />
                        </button>
                    </div>
                    <ul className={styles.submenuList}>
                        {openMenuItem.children.map(child => {
                            const isActive = isItemActive(child.href);

                            return (
                                <li key={child.title}>
                                    <Link
                                        href={child.href}
                                        onClick={() => setOpenSubMenu(null)}
                                        className={clsx(styles.submenuLink, {
                                            [styles.submenuLinkActive]: isActive,
                                        })}
                                    >
                                        {child.title}
                                    </Link>
                                </li>
                            );
                        })}
                    </ul>
                </div>
            )}
        </aside>
    );
};
