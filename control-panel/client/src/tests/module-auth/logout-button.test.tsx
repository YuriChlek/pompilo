import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useRouter } from 'next/navigation';
import { useLogout } from '@/features/module-auth/hooks/mutation';
import { LogoutButton } from '@/features/module-auth/components/logout-button/logout-button';

vi.mock('next/navigation', () => ({
    useRouter: vi.fn(),
}));

vi.mock('@/features/module-auth/hooks/mutation', () => ({
    useLogout: vi.fn(),
}));

describe('LogoutButton', () => {
    const mockPush = vi.fn();
    const mockMutate = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useRouter).mockReturnValue({
            back: vi.fn(),
            forward: vi.fn(),
            prefetch: vi.fn(),
            push: mockPush,
            refresh: vi.fn(),
            replace: vi.fn(),
        });
        vi.mocked(useLogout).mockReturnValue({
            mutate: mockMutate,
            isPending: false,
        } as unknown as ReturnType<typeof useLogout>);
    });

    it('renders login label when not authenticated', () => {
        render(<LogoutButton isAuthenticated={false} />);
        expect(screen.getByText('Log in')).toBeDefined();
    });

    it('renders logout label when authenticated', () => {
        render(<LogoutButton isAuthenticated={true} />);
        expect(screen.getByText('Log out')).toBeDefined();
    });

    it('hides label when forceCompact is true', () => {
        render(<LogoutButton isAuthenticated={true} forceCompact={true} />);
        expect(screen.queryByText('Log out')).toBeNull();
    });

    it('calls router.push when clicking login', () => {
        render(<LogoutButton isAuthenticated={false} loginPath="/custom-login" />);
        fireEvent.click(screen.getByRole('button'));
        expect(mockPush).toHaveBeenCalledWith('/custom-login');
    });

    it('calls mutate when clicking logout', () => {
        render(<LogoutButton isAuthenticated={true} />);
        fireEvent.click(screen.getByRole('button'));
        expect(mockMutate).toHaveBeenCalled();
    });
});
