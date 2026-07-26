import clsx from 'clsx';
import type { CSSProperties } from 'react';
import styles from '@/components/loader/styles.module.css';

type LoaderSize = 'sm' | 'md' | 'lg';

type LoaderProps = {
    size?: LoaderSize;
    label?: string;
    className?: string;
};

const SIZE_MAP: Record<LoaderSize, string> = {
    sm: '1.5rem',
    md: '2.5rem',
    lg: '3.25rem',
};

export const Loader = ({ size = 'md', label = 'Loading', className }: LoaderProps) => {
    const style = {
        '--loader-size': SIZE_MAP[size],
    } as CSSProperties;

    return (
        <div
            className={clsx(styles.loader, className)}
            role="status"
            aria-live="polite"
            aria-label={label}
            style={style}
        >
            <span className={styles.spinner} aria-hidden="true" />
            {label ? <span className={styles.label}>{label}</span> : null}
        </div>
    );
};
