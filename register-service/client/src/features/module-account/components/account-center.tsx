'use client';

import { ReactNode, Suspense } from 'react';
import { useSearchParams, useRouter, usePathname } from 'next/navigation';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
    faShieldHalved,
    faMobileScreen,
    faLock,
    faTriangleExclamation,
    faChevronDown,
    faChevronUp,
} from '@fortawesome/free-solid-svg-icons';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import styles from './styles.module.css';
import { PageTitle } from '@/components/page-title/page-title';

// Import settings sections
import { SecuritySettingsSection } from './security-settings-section';
import { SessionsSettingsSection } from './sessions-settings-section';
import { PrivacySettingsSection } from './privacy-settings-section';
import { DangerZoneSection } from './danger-zone-section';

type AccountCenterProps = {
    title: string;
    description: string;
    children?: ReactNode;
};

const TABS = [
    { id: 'security', label: 'Безпека', icon: faShieldHalved },
    { id: 'sessions', label: 'Пристрої', icon: faMobileScreen },
    { id: 'privacy', label: 'Приватність', icon: faLock },
    { id: 'danger', label: 'Видалення', icon: faTriangleExclamation, isDanger: true },
];

const AccountCenterInner = ({ title, description, children }: AccountCenterProps) => {
    const searchParams = useSearchParams();
    const router = useRouter();
    const pathname = usePathname();

    const requestedTab = searchParams.get('tab') || 'security';
    const activeTab = TABS.some(tab => tab.id === requestedTab) ? requestedTab : 'security';

    const handleTabChange = (tabId: string) => {
        const params = new URLSearchParams(searchParams.toString());
        params.set('tab', tabId);
        router.push(`${pathname}?${params.toString()}`);
    };

    const renderSectionContent = (tabId: string) => {
        switch (tabId) {
            case 'security':
                return <SecuritySettingsSection role={UserRoles.USER} />;
            case 'sessions':
                return <SessionsSettingsSection role={UserRoles.USER} />;
            case 'privacy':
                return <PrivacySettingsSection />;
            case 'danger':
                return <DangerZoneSection role={UserRoles.USER} />;
            default:
                return children ?? <SecuritySettingsSection role={UserRoles.USER} />;
        }
    };

    return (
        <div className={styles.wrapper}>
            <div className={styles.intro}>
                <PageTitle pageTitle={title} />
                <p className={styles.description}>{description}</p>
            </div>

            {/* Desktop View (Sidebar + Detail Content) */}
            <div className={styles.desktopLayout}>
                <aside className={styles.sidebar}>
                    {TABS.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => handleTabChange(tab.id)}
                            className={`${styles.sidebarItem} ${activeTab === tab.id ? styles.activeSidebarItem : ''} ${tab.isDanger ? styles.dangerSidebarItem : ''}`}
                        >
                            <FontAwesomeIcon icon={tab.icon} className={styles.sidebarIcon} />
                            <span>{tab.label}</span>
                        </button>
                    ))}
                </aside>

                <main className={styles.detailContainer}>
                    <div className={styles.detailCard}>
                        {renderSectionContent(activeTab)}
                    </div>
                </main>
            </div>

            {/* Mobile View (Accordions) */}
            <div className={styles.mobileLayout}>
                {TABS.map(tab => {
                    const isOpen = activeTab === tab.id;
                    return (
                        <div
                            key={tab.id}
                            className={`${styles.accordionItem} ${isOpen ? styles.openAccordionItem : ''} ${tab.isDanger ? styles.dangerAccordionItem : ''}`}
                        >
                            <button
                                onClick={() => handleTabChange(isOpen ? 'security' : tab.id)}
                                className={styles.accordionHeader}
                            >
                                <span className={styles.accordionHeaderLeft}>
                                    <FontAwesomeIcon icon={tab.icon} className={styles.accordionIcon} />
                                    <span className={styles.accordionTitle}>{tab.label}</span>
                                </span>
                                <FontAwesomeIcon icon={isOpen ? faChevronUp : faChevronDown} className={styles.accordionChevron} />
                            </button>
                            {isOpen && (
                                <div className={styles.accordionContent}>
                                    {renderSectionContent(tab.id)}
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export const AccountCenter = (props: AccountCenterProps) => {
    return (
        <Suspense fallback={<div className={styles.loading}>Завантаження...</div>}>
            <AccountCenterInner {...props} />
        </Suspense>
    );
};

type AccountSectionProps = {
    title?: string;
    description?: string;
    children: ReactNode;
};

export const AccountSection = ({ title, description, children }: AccountSectionProps) => {
    return (
        <div className={styles.section}>
            {(title || description) && (
                <div className={styles.sectionHeader}>
                    {title && <h3 className={styles.sectionTitle}>{title}</h3>}
                    {description && <p className={styles.sectionDescription}>{description}</p>}
                </div>
            )}
            <div className={styles.sectionContent}>
                {children}
            </div>
        </div>
    );
};
