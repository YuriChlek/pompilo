'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createContext, useContext, useState, useSyncExternalStore, type ReactNode } from 'react';
import { QueryDevtools } from '@/lib/providers/query-devtools';
import { isTheme } from '@/shared/theme/lib/is-theme';
import type { Theme } from '@/shared/theme/types/theme.types';

const ThemeContext = createContext<Theme>('light');
const THEME_CHANGE_EVENT = 'themechange';

function subscribeToTheme(listener: () => void) {
    window.addEventListener(THEME_CHANGE_EVENT, listener);

    return () => window.removeEventListener(THEME_CHANGE_EVENT, listener);
}

function getDocumentTheme(): Theme {
    const documentTheme = document.documentElement.dataset.theme;

    return isTheme(documentTheme) ? documentTheme : 'light';
}

function getServerTheme(): Theme {
    return 'light';
}

export function notifyThemeChange() {
    window.dispatchEvent(new Event(THEME_CHANGE_EVENT));
}

export function useInitialTheme() {
    return useContext(ThemeContext);
}

export function Providers({ children }: Readonly<{ children: ReactNode }>) {
    const [queryClient] = useState(() => new QueryClient());
    const theme = useSyncExternalStore(subscribeToTheme, getDocumentTheme, getServerTheme);

    return (
        <ThemeContext.Provider value={theme}>
            <QueryClientProvider client={queryClient}>
                {children}
                <QueryDevtools />
            </QueryClientProvider>
        </ThemeContext.Provider>
    );
}
