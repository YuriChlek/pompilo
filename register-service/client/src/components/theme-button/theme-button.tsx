'use client';

import { useTransition } from 'react';
import styles from '@/components/theme-button/styles.module.css';
import type { Theme } from '@/shared/theme/types/theme.types';
import { persistTheme } from '@/shared/theme/actions';
import { notifyThemeChange, useInitialTheme } from '@/lib/providers/providers';

export const ThemeButton = () => {
    const theme = useInitialTheme();
    const [isPending, startTransition] = useTransition();

    const handleToggle = async () => {
        const updatedTheme: Theme = theme === 'dark' ? 'light' : 'dark';

        document.documentElement.dataset.theme = updatedTheme;
        notifyThemeChange();

        startTransition(async () => {
            await persistTheme(updatedTheme);
        });
    };

    const nextThemeLabel = theme === 'light' ? 'Dark mode' : 'Light mode';
    const isLightTheme = theme === 'light';

    return (
        <button
            className={styles.toggleThemeButton}
            type="button"
            onClick={handleToggle}
            aria-label={`Switch to ${nextThemeLabel.toLowerCase()}`}
            aria-pressed={!isLightTheme}
            title={`Switch to ${nextThemeLabel.toLowerCase()}`}
            disabled={isPending}
        >
            <span className={styles.track} aria-hidden="true">
                <span className={styles.icon}>{isLightTheme ? '☀' : '☾'}</span>
            </span>
            <span className={styles.label}>{nextThemeLabel}</span>
        </button>
    );
};
