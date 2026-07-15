'use client';

import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import { SecuritySettingsSection } from '@/features/module-account/components/security-settings-section';
import { SessionsSettingsSection } from '@/features/module-account/components/sessions-settings-section';
import { DangerZoneSection } from '@/features/module-account/components/danger-zone-section';
import { CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM } from '@/features/module-menu/config/menu.config';
import styles from './styles.module.css';

export const AccountSecurityPage = () => {
    const role = UserRoles.USER;

    return (
        <div className={styles.wrapper}>
            <div className={styles.intro}>
                <h1 className={styles.sectionTitle}>{CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM.title}</h1>
                <p className={styles.description}>
                    Manage login credentials, active sessions and account lifecycle.
                </p>
            </div>

            <main className={styles.sections}>
                <div className={styles.detailCard}>
                    <SecuritySettingsSection role={role} />
                </div>
                <div className={styles.detailCard}>
                    <SessionsSettingsSection role={role} />
                </div>
                <div className={styles.detailCard}>
                    <DangerZoneSection role={role} />
                </div>
            </main>
        </div>
    );
};
