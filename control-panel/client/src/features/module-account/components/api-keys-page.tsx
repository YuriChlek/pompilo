'use client';

import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
    faCircleInfo,
    faKey,
    faLock,
    faPlus,
    faRotate,
    faTrash,
} from '@fortawesome/free-solid-svg-icons';
import { CUSTOMER_API_KEYS_MENU_ITEM } from '@/features/module-menu/config/menu.config';
import styles from './api-keys-page.module.css';

const scopes = ['identity:read', 'sessions:read', 'security-events:read'];

export const ApiKeysPage = () => {
    return (
        <div className={styles.wrapper}>
            <header className={styles.header}>
                <div>
                    <h1 className={styles.title}>{CUSTOMER_API_KEYS_MENU_ITEM.title}</h1>
                    <p className={styles.description}>
                        Manage programmatic access credentials for this account.
                    </p>
                </div>
                <button className={styles.primaryButton} type="button" disabled>
                    <FontAwesomeIcon icon={faPlus} />
                    <span>Create key</span>
                </button>
            </header>

            <main className={styles.content}>
                <section className={styles.panel} aria-labelledby="api-keys-list-title">
                    <div className={styles.panelHeader}>
                        <div className={styles.panelTitleGroup}>
                            <FontAwesomeIcon icon={faKey} className={styles.panelIcon} />
                            <div>
                                <h2 id="api-keys-list-title" className={styles.panelTitle}>
                                    Active keys
                                </h2>
                                <p className={styles.panelDescription}>
                                    No active API keys are attached to this account.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className={styles.emptyState}>
                        <FontAwesomeIcon icon={faLock} className={styles.emptyIcon} />
                        <div>
                            <h3 className={styles.emptyTitle}>No API keys</h3>
                            <p className={styles.emptyText}>
                                Key creation requires the API key backend to be enabled.
                            </p>
                        </div>
                    </div>
                </section>

                <section className={styles.panel} aria-labelledby="api-key-controls-title">
                    <div className={styles.panelHeader}>
                        <div className={styles.panelTitleGroup}>
                            <FontAwesomeIcon icon={faCircleInfo} className={styles.panelIcon} />
                            <div>
                                <h2 id="api-key-controls-title" className={styles.panelTitle}>
                                    Key policy
                                </h2>
                                <p className={styles.panelDescription}>
                                    Keys will use explicit scopes and account-bound ownership.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className={styles.policyGrid}>
                        <div className={styles.policyBlock}>
                            <span className={styles.policyLabel}>Available scopes</span>
                            <div className={styles.scopeList}>
                                {scopes.map(scope => (
                                    <span className={styles.scopePill} key={scope}>
                                        {scope}
                                    </span>
                                ))}
                            </div>
                        </div>

                        <div className={styles.actionPreview} aria-hidden="true">
                            <button className={styles.iconButton} type="button" disabled>
                                <FontAwesomeIcon icon={faRotate} />
                            </button>
                            <button className={styles.iconButtonDanger} type="button" disabled>
                                <FontAwesomeIcon icon={faTrash} />
                            </button>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
};
