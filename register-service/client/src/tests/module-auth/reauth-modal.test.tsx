import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { authService } from '@/features/module-auth/api-service/client';
import { ReauthModal } from '@/features/module-auth/components/reauth-modal/reauth-modal';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';

vi.mock('@/features/module-auth/api-service/client', () => ({
    authService: {
        reauth: vi.fn(),
    },
}));

describe('ReauthModal', () => {
    const mockClose = vi.fn();
    const mockSuccess = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('renders nothing when isOpen is false', () => {
        const { container } = render(
            <ReauthModal
                isOpen={false}
                onClose={mockClose}
                onSuccess={mockSuccess}
                actionScope="password_change"
                role={UserRoles.USER}
            />
        );
        expect(container.firstChild).toBeNull();
    });

    it('renders heading and fields when isOpen is true', () => {
        render(
            <ReauthModal
                isOpen={true}
                onClose={mockClose}
                onSuccess={mockSuccess}
                actionScope="password_change"
                role={UserRoles.USER}
            />
        );

        expect(screen.getByText('Потрібне підтвердження')).toBeDefined();
        expect(screen.getByPlaceholderText('Введіть ваш пароль')).toBeDefined();
        expect(screen.getByRole('button', { name: 'Підтвердити' })).toBeDefined();
    });

    it('calls authService.reauth and triggers onSuccess on successful form submission', async () => {
        vi.mocked(authService.reauth).mockResolvedValue({
            confirmationToken: 'reauth-token-xyz',
            expiresAt: '2026-01-01T00:00:00.000Z',
        });

        render(
            <ReauthModal
                isOpen={true}
                onClose={mockClose}
                onSuccess={mockSuccess}
                actionScope="password_change"
                role={UserRoles.USER}
            />
        );

        const passwordInput = screen.getByPlaceholderText('Введіть ваш пароль');
        fireEvent.change(passwordInput, { target: { value: 'my-secret-pass' } });

        const submitBtn = screen.getByRole('button', { name: 'Підтвердити' });
        fireEvent.click(submitBtn);

        await waitFor(() => {
            expect(authService.reauth).toHaveBeenCalledWith(
                'my-secret-pass',
                'password_change',
                UserRoles.USER
            );
            expect(mockSuccess).toHaveBeenCalledWith('reauth-token-xyz');
            expect(mockClose).toHaveBeenCalled();
        });
    });

    it('displays error message on failed reauth', async () => {
        vi.mocked(authService.reauth).mockRejectedValue(new Error('Невірний пароль'));

        render(
            <ReauthModal
                isOpen={true}
                onClose={mockClose}
                onSuccess={mockSuccess}
                actionScope="password_change"
                role={UserRoles.USER}
            />
        );

        const passwordInput = screen.getByPlaceholderText('Введіть ваш пароль');
        fireEvent.change(passwordInput, { target: { value: 'wrong-pass' } });

        fireEvent.click(screen.getByRole('button', { name: 'Підтвердити' }));

        await waitFor(() => {
            expect(screen.getByText('Невірний пароль')).toBeDefined();
            expect(mockSuccess).not.toHaveBeenCalled();
            expect(mockClose).not.toHaveBeenCalled();
        });
    });

    it('triggers onClose when cancel button is clicked', () => {
        render(
            <ReauthModal
                isOpen={true}
                onClose={mockClose}
                onSuccess={mockSuccess}
                actionScope="password_change"
                role={UserRoles.USER}
            />
        );

        fireEvent.click(screen.getByRole('button', { name: 'Скасувати' }));
        expect(mockClose).toHaveBeenCalled();
    });
});
