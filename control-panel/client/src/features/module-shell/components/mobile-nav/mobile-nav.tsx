'use client';

import clsx from 'clsx';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { getMenu } from '@/features/module-menu/config/menu.config';
import { MenuTypes } from '@/features/module-menu/enums/menu.enums';
import styles from './styles.module.css';
import type { MenuItem } from '@/features/module-menu/interfaces/menu.interfaces';

interface MobileNavProps {
    menuType: MenuTypes;
}

export const MobileNav = ({ menuType }: MobileNavProps) => {
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

    // Filter items to show on mobile (limit to 5 as per plan if needed, but let's show all for now with overflow)
    const mobileItems = menuItems.filter(item => item.mobile !== false);

    return (
        <nav className={styles.mobileNav}>
            {mobileItems.map(item => {
                const isActive = isItemActive(item);

                if (item.disabled) {
                    return (
                        <div key={item.title} className={clsx(styles.navItem, styles.disabled)}>
                            <div className={styles.iconWrapper}>
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
                            {item.icon && typeof item.icon !== 'string' && (
                                <FontAwesomeIcon icon={item.icon} />
                            )}
                        </div>
                        <span className={styles.label}>{item.title}</span>
                    </Link>
                );
            })}
        </nav>
    );
};
