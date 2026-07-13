import Link from 'next/link';
import styles from '@/features/module-auth/components/user-badge/styles.module.css';
import type { UserBadgeProps } from '@/features/module-auth/interfaces/component-props.interfaces';
import { CUSTOMER_DEFAULT_MENU_ITEM } from '@/features/module-menu/config/menu.config';

export const UserBadge = ({ userName, href = CUSTOMER_DEFAULT_MENU_ITEM.href }: UserBadgeProps) => {
    return (
        <Link className={styles.userBadge} href={href}>
            {userName}
        </Link>
    );
};
