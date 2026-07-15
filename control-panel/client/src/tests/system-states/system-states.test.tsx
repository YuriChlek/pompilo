import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { AppLoadingSkeleton } from '@/components/app-loading-skeleton/app-loading-skeleton';
import { SystemState } from '@/components/system-state/system-state';

describe('system states', () => {
    it('renders a recovery action and calls the retry handler', () => {
        const onRetry = vi.fn();

        render(
            <SystemState
                eyebrow="Application error"
                title="Something went wrong"
                description="Try again."
                tone="error"
                onRetry={onRetry}
            />,
        );

        fireEvent.click(screen.getByRole('button', { name: 'Try again' }));

        expect(onRetry).toHaveBeenCalledTimes(1);
    });

    it('provides homepage navigation for non-recoverable states', () => {
        render(
            <SystemState
                eyebrow="Page not found"
                title="This route is off the map"
                description="The page could not be found."
                tone="not-found"
            />,
        );

        expect(screen.getByRole('link', { name: 'Go to homepage' })).toHaveAttribute('href', '/');
    });

    it('exposes an accessible loading status', () => {
        render(<AppLoadingSkeleton />);

        expect(screen.getByRole('status', { name: 'Loading page' })).toBeInTheDocument();
    });
});
