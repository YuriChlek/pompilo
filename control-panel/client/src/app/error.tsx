'use client';

import { useEffect } from 'react';
import { SystemState } from '@/components/system-state/system-state';

interface ErrorPageProps {
    error: Error & { digest?: string };
    reset: () => void;
}

export default function ErrorPage({ error, reset }: ErrorPageProps) {
    useEffect(() => {
        console.error(error);
    }, [error]);

    return (
        <SystemState
            eyebrow="Application error"
            title="Something went wrong"
            description="We could not load this part of Pampilo. Try the request again or return to the homepage."
            tone="error"
            onRetry={reset}
        />
    );
}
