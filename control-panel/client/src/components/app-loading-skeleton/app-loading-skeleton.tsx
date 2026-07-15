import styles from '@/components/app-loading-skeleton/styles.module.css';

export function AppLoadingSkeleton() {
    return (
        <div className={styles.page} role="status" aria-live="polite" aria-label="Loading page">
            <aside className={styles.sidebar} aria-hidden="true">
                <div className={`${styles.skeleton} ${styles.logo}`} />
                <div className={styles.navigation}>
                    {Array.from({ length: 6 }, (_, index) => (
                        <div className={styles.navItem} key={index}>
                            <div className={`${styles.skeleton} ${styles.navIcon}`} />
                            <div className={`${styles.skeleton} ${styles.navLabel}`} />
                        </div>
                    ))}
                </div>
                <div className={`${styles.skeleton} ${styles.profile}`} />
            </aside>

            <main className={styles.main}>
                <div className={styles.content}>
                    <div className={styles.heading}>
                        <div className={`${styles.skeleton} ${styles.eyebrow}`} />
                        <div className={`${styles.skeleton} ${styles.title}`} />
                        <div className={`${styles.skeleton} ${styles.subtitle}`} />
                    </div>

                    <div className={styles.grid}>
                        {Array.from({ length: 3 }, (_, index) => (
                            <div className={styles.card} key={index}>
                                <div className={`${styles.skeleton} ${styles.cardIcon}`} />
                                <div className={`${styles.skeleton} ${styles.cardTitle}`} />
                                <div className={`${styles.skeleton} ${styles.cardLine}`} />
                                <div
                                    className={`${styles.skeleton} ${styles.cardLine} ${styles.cardLineShort}`}
                                />
                            </div>
                        ))}
                    </div>

                    <div className={styles.panel}>
                        <div className={`${styles.skeleton} ${styles.panelTitle}`} />
                        <div className={styles.rows}>
                            {Array.from({ length: 4 }, (_, index) => (
                                <div className={styles.row} key={index}>
                                    <div className={`${styles.skeleton} ${styles.avatar}`} />
                                    <div className={styles.rowCopy}>
                                        <div className={`${styles.skeleton} ${styles.rowTitle}`} />
                                        <div className={`${styles.skeleton} ${styles.rowLine}`} />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </main>

            <span className={styles.srOnly}>Loading page content</span>
        </div>
    );
}
