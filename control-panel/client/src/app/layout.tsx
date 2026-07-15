import type { Metadata } from 'next';
import { config } from '@fortawesome/fontawesome-svg-core';
import '@fortawesome/fontawesome-svg-core/styles.css';
import { Providers } from '@/lib/providers/providers';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v16-appRouter';
import '../shared/css/globals.css';
import '../shared/css/normalise.css';

config.autoAddCss = false;

const themeBootstrapScript = `
(() => {
    const match = document.cookie.match(/(?:^|; )theme=(light|dark)(?:;|$)/);
    document.documentElement.dataset.theme = match?.[1] ?? 'light';
})();
`;

export const metadata: Metadata = {
    title: 'Identity Service',
    description: 'Identity and registration workspace for the trading platform.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
    return (
        <html
            lang="en"
            data-theme="light"
            suppressHydrationWarning
        >
            <head>
                <script
                    id="theme-bootstrap"
                    dangerouslySetInnerHTML={{ __html: themeBootstrapScript }}
                />
            </head>
            <body>
                <AppRouterCacheProvider>
                    <Providers>{children}</Providers>
                </AppRouterCacheProvider>
            </body>
        </html>
    );
}
