'use client';

import { useEffect } from 'react';
import { SystemState } from '@/components/system-state/system-state';
import '../shared/css/globals.css';
import '../shared/css/normalise.css';

interface GlobalErrorProps {
    error: Error & { digest?: string };
    reset: () => void;
}

export default function GlobalError({ error, reset }: GlobalErrorProps) {
    useEffect(() => {
        console.error(error);
    }, [error]);

    return (
        <html lang="en" data-theme="light">
            <body>
                <SystemState
                    eyebrow="Critical error"
                    title="Pampilo is temporarily unavailable"
                    description="The application could not start correctly. Try again, and if the problem continues, return later."
                    tone="error"
                    retryLabel="Reload application"
                    onRetry={reset}
                />
            </body>
        </html>
    );
}
