import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import {
    useLogoutAllSessionsMutation,
    useRevokeSessionMutation,
} from '@/features/module-account/hooks/mutation';
import { ACCOUNT_QUERY_KEYS } from '@/features/module-account/hooks/query';
import { accountApiService } from '@/features/module-account/api-service/client';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';

vi.mock('@/features/module-account/api-service/client', () => ({
    accountApiService: {
        logoutAllSessions: vi.fn(),
        revokeSession: vi.fn(),
    },
}));

describe('useRevokeSessionMutation - Revoke Current Session (Phase 29.3)', () => {
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

    it('redirects to /login and clears query cache if current session is revoked', async () => {
        const sessions = [
            {
                id: 'current-sess-1',
                ipAddress: '127.0.0.1',
                userAgent: 'macintosh',
                createdAt: '2026-06-05T00:00:00Z',
                currentSession: true,
            },
            {
                id: 'other-sess-2',
                ipAddress: '192.168.1.1',
                userAgent: 'iphone',
                createdAt: '2026-06-05T00:00:00Z',
                currentSession: false,
            },
        ];

        queryClient.setQueryData(ACCOUNT_QUERY_KEYS.sessions(UserRoles.USER), sessions);

        const clearSpy = vi.spyOn(queryClient, 'clear');

        const originalLocation = window.location;
        const locationMock = {
            ...originalLocation,
            href: 'http://localhost:3000/account/security',
        };
        vi.stubGlobal('location', locationMock);

        vi.mocked(accountApiService.revokeSession).mockResolvedValueOnce();

        const { result } = renderHook(() => useRevokeSessionMutation(UserRoles.USER), { wrapper });

        await act(async () => {
            await result.current.mutateAsync('current-sess-1');
        });

        expect(accountApiService.revokeSession).toHaveBeenCalledWith(UserRoles.USER, 'current-sess-1');
        expect(clearSpy).toHaveBeenCalled();
        expect(locationMock.href).toBe('/login');
    });

    it('redirects to /admin/login and clears query cache if current admin session is revoked', async () => {
        const sessions = [
            {
                id: 'admin-current-sess-1',
                ipAddress: '127.0.0.1',
                userAgent: 'macintosh',
                createdAt: '2026-06-05T00:00:00Z',
                currentSession: true,
            },
        ];

        queryClient.setQueryData(ACCOUNT_QUERY_KEYS.sessions(UserRoles.PLATFORM_ADMIN), sessions);

        const clearSpy = vi.spyOn(queryClient, 'clear');

        const originalLocation = window.location;
        const locationMock = {
            ...originalLocation,
            href: 'http://localhost:3000/admin/settings',
        };
        vi.stubGlobal('location', locationMock);

        vi.mocked(accountApiService.revokeSession).mockResolvedValueOnce();

        const { result } = renderHook(() => useRevokeSessionMutation(UserRoles.PLATFORM_ADMIN), { wrapper });

        await act(async () => {
            await result.current.mutateAsync('admin-current-sess-1');
        });

        expect(accountApiService.revokeSession).toHaveBeenCalledWith(UserRoles.PLATFORM_ADMIN, 'admin-current-sess-1');
        expect(clearSpy).toHaveBeenCalled();
        expect(locationMock.href).toBe('/admin/login');
    });

    it('only invalidates queries and does not redirect if another session is revoked', async () => {
        const sessions = [
            {
                id: 'current-sess-1',
                ipAddress: '127.0.0.1',
                userAgent: 'macintosh',
                createdAt: '2026-06-05T00:00:00Z',
                currentSession: true,
            },
            {
                id: 'other-sess-2',
                ipAddress: '192.168.1.1',
                userAgent: 'iphone',
                createdAt: '2026-06-05T00:00:00Z',
                currentSession: false,
            },
        ];

        queryClient.setQueryData(ACCOUNT_QUERY_KEYS.sessions(UserRoles.USER), sessions);

        const clearSpy = vi.spyOn(queryClient, 'clear');
        const invalidateSpy = vi.spyOn(queryClient, 'invalidateQueries');

        const originalLocation = window.location;
        const locationMock = {
            ...originalLocation,
            href: 'http://localhost:3000/account/security',
        };
        vi.stubGlobal('location', locationMock);

        vi.mocked(accountApiService.revokeSession).mockResolvedValueOnce();

        const { result } = renderHook(() => useRevokeSessionMutation(UserRoles.USER), { wrapper });

        await act(async () => {
            await result.current.mutateAsync('other-sess-2');
        });

        expect(accountApiService.revokeSession).toHaveBeenCalledWith(UserRoles.USER, 'other-sess-2');
        expect(clearSpy).not.toHaveBeenCalled();
        expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ACCOUNT_QUERY_KEYS.sessions(UserRoles.USER) });
        expect(locationMock.href).toBe('http://localhost:3000/account/security');
    });
});

describe('useLogoutAllSessionsMutation - Logout All Sessions (Phase 29.4)', () => {
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

    it('clears query cache and redirects to /login after user logout-all succeeds', async () => {
        const clearSpy = vi.spyOn(queryClient, 'clear');
        const originalLocation = window.location;
        const locationMock = {
            ...originalLocation,
            href: 'http://localhost:3000/account/security',
        };
        vi.stubGlobal('location', locationMock);

        vi.mocked(accountApiService.logoutAllSessions).mockResolvedValueOnce();

        const { result } = renderHook(() => useLogoutAllSessionsMutation(UserRoles.USER), { wrapper });

        await act(async () => {
            await result.current.mutateAsync();
        });

        expect(accountApiService.logoutAllSessions).toHaveBeenCalledWith(UserRoles.USER);
        expect(clearSpy).toHaveBeenCalled();
        expect(locationMock.href).toBe('/login');
    });

    it('clears query cache and redirects to /admin/login after admin logout-all succeeds', async () => {
        const clearSpy = vi.spyOn(queryClient, 'clear');
        const originalLocation = window.location;
        const locationMock = {
            ...originalLocation,
            href: 'http://localhost:3000/admin/settings',
        };
        vi.stubGlobal('location', locationMock);

        vi.mocked(accountApiService.logoutAllSessions).mockResolvedValueOnce();

        const { result } = renderHook(() => useLogoutAllSessionsMutation(UserRoles.PLATFORM_ADMIN), { wrapper });

        await act(async () => {
            await result.current.mutateAsync();
        });

        expect(accountApiService.logoutAllSessions).toHaveBeenCalledWith(UserRoles.PLATFORM_ADMIN);
        expect(clearSpy).toHaveBeenCalled();
        expect(locationMock.href).toBe('/admin/login');
    });
});
