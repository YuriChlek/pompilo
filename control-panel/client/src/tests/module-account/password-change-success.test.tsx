import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import { useChangePasswordMutation } from '@/features/module-account/hooks/mutation';
import { accountApiService } from '@/features/module-account/api-service/client';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';

vi.mock('@/features/module-account/api-service/client', () => ({
    accountApiService: {
        changePassword: vi.fn(),
    },
}));

describe('useChangePasswordMutation - Password Change Success Flow (Phase 31.4)', () => {
    let queryClient: QueryClient;

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient = new QueryClient({
            defaultOptions: {
                queries: {
                    retry: false,
                },
            },
        });
    });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );

    it('clears query cache, expires cookies, and redirects to /login on user password change success', async () => {
        const clearSpy = vi.spyOn(queryClient, 'clear');

        const originalLocation = window.location;
        const locationMock = {
            ...originalLocation,
            href: 'http://localhost:3000/account/security',
        };
        vi.stubGlobal('location', locationMock);

        const cookieSetSpy = vi.fn();
        const originalCookieDescriptor = Object.getOwnPropertyDescriptor(Document.prototype, 'cookie') || 
                                         Object.getOwnPropertyDescriptor(document, 'cookie');

        Object.defineProperty(document, 'cookie', {
            configurable: true,
            set: cookieSetSpy,
            get: () => '',
        });

        vi.mocked(accountApiService.changePassword).mockResolvedValueOnce();

        const { result } = renderHook(() => useChangePasswordMutation(UserRoles.USER), { wrapper });

        await act(async () => {
            await result.current.mutateAsync({
                data: { oldPassword: 'old', newPassword: 'new' },
                reauthConfirmationToken: 'token-abc',
            });
        });

        expect(accountApiService.changePassword).toHaveBeenCalledWith(
            UserRoles.USER,
            { oldPassword: 'old', newPassword: 'new' },
            'token-abc'
        );
        expect(clearSpy).toHaveBeenCalled();
        expect(locationMock.href).toBe('/login');

        expect(cookieSetSpy).toHaveBeenCalledWith(
            expect.stringContaining('customerAccessToken=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT')
        );
        expect(cookieSetSpy).toHaveBeenCalledWith(
            expect.stringContaining('customerRefreshToken=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT')
        );

        // Restore original descriptor
        if (originalCookieDescriptor) {
            Object.defineProperty(document, 'cookie', originalCookieDescriptor);
        }
    });

    it('redirects to /admin/login on admin password change success', async () => {
        const originalLocation = window.location;
        const locationMock = {
            ...originalLocation,
            href: 'http://localhost:3000/admin/settings',
        };
        vi.stubGlobal('location', locationMock);

        vi.mocked(accountApiService.changePassword).mockResolvedValueOnce();

        const { result } = renderHook(() => useChangePasswordMutation(UserRoles.PLATFORM_ADMIN), { wrapper });

        await act(async () => {
            await result.current.mutateAsync({
                data: { oldPassword: 'old', newPassword: 'new' },
                reauthConfirmationToken: 'token-abc',
            });
        });

        expect(locationMock.href).toBe('/admin/login');
    });
});
