'use client';

import Link from 'next/link';
import { Button } from '@/components/button/button';
import { Logo } from '@/components/logo/logo';
import styles from '@/components/system-state/styles.module.css';

type SystemStateTone = 'error' | 'not-found';

interface SystemStateProps {
    eyebrow: string;
    title: string;
    description: string;
    tone: SystemStateTone;
    retryLabel?: string;
    onRetry?: () => void;
}

export function SystemState({
    eyebrow,
    title,
    description,
    tone,
    retryLabel = 'Try again',
    onRetry,
}: SystemStateProps) {
    return (
        <div className={styles.page}>
            <div className={styles.ambient} aria-hidden="true" />

            <section className={styles.card} aria-labelledby="system-state-title">
                <div className={styles.logo}>
                    <Logo />
                </div>

                <div className={styles.content}>
                    <div className={`${styles.statusMark} ${styles[tone]}`} aria-hidden="true">
                        <span>{tone === 'not-found' ? '404' : '!'}</span>
                    </div>

                    <p className={styles.eyebrow}>{eyebrow}</p>
                    <h1 id="system-state-title" className={styles.title}>
                        {title}
                    </h1>
                    <p className={styles.description}>{description}</p>
                </div>

                <div className={styles.actions}>
                    {onRetry ? (
                        <Button onClick={onRetry}>{retryLabel}</Button>
                    ) : null}
                    <Link className={styles.homeLink} href="/">
                        Go to homepage
                    </Link>
                </div>
            </section>
        </div>
    );
}
