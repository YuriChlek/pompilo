import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import * as React from 'react';
import { useResendCheckpoint } from '@/features/module-auth/hooks/mutation';
import { authService } from '@/features/module-auth/api-service/client';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';

vi.mock('@/features/module-auth/api-service/client', () => ({
    authService: {
        resendCheckpoint: vi.fn(),
    },
}));

vi.mock('next/navigation', () => ({
    useRouter: () => ({
        push: vi.fn(),
        replace: vi.fn(),
        refresh: vi.fn(),
    }),
}));

describe('useResendCheckpoint', () => {
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

    it('returns the replacement checkpoint response from the auth API service', async () => {
        vi.mocked(authService.resendCheckpoint).mockResolvedValue({
            checkpointRequired: true,
            loginChallengeId: 'new-challenge-id',
            checkpointToken: 'new-checkpoint-token',
            expiresInSeconds: 300,
            resendAvailableInSeconds: 60,
        });

        const { result } = renderHook(() => useResendCheckpoint(), { wrapper });

        await act(async () => {
            const response = await result.current.mutateAsync({
                checkpointToken: 'old-checkpoint-token',
                role: UserRoles.PLATFORM_ADMIN,
            });

            expect(response).toMatchObject({
                loginChallengeId: 'new-challenge-id',
                checkpointToken: 'new-checkpoint-token',
            });
        });

        expect(authService.resendCheckpoint).toHaveBeenCalledWith(
            'old-checkpoint-token',
            UserRoles.PLATFORM_ADMIN,
        );
    });
});
