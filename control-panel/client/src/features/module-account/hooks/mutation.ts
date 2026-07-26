import { useMutation, useQueryClient } from '@tanstack/react-query';
import { accountApiService } from '@/features/module-account/api-service/client';
import { ACCOUNT_QUERY_KEYS } from '@/features/module-account/hooks/query';
import { UserRoles, COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';
import { UserSession } from '@/features/module-account/interfaces/account.interfaces';

export const useRevokeSessionMutation = (role: UserRoles) => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (sessionId: string) => accountApiService.revokeSession(role, sessionId),
        onSuccess: (_, sessionId) => {
            const sessions = queryClient.getQueryData<UserSession[]>(ACCOUNT_QUERY_KEYS.sessions(role));
            const currentSession = sessions?.find(s => s.currentSession);

            if (currentSession && currentSession.id === sessionId) {
                queryClient.clear();
                if (typeof window !== 'undefined') {
                    window.location.href =
                        role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN
                            ? '/admin/login'
                            : '/login';
                }
            } else {
                queryClient.invalidateQueries({ queryKey: ACCOUNT_QUERY_KEYS.sessions(role) });
            }
        },
    });
};

export const useRevokeOtherSessionsMutation = (role: UserRoles) => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (reauthConfirmationToken?: string) =>
            accountApiService.revokeOtherSessions(role, reauthConfirmationToken),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ACCOUNT_QUERY_KEYS.sessions(role) });
        },
    });
};

export const useLogoutAllSessionsMutation = (role: UserRoles) => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: () => accountApiService.logoutAllSessions(role),
        onSuccess: () => {
            queryClient.clear();
            if (typeof window !== 'undefined') {
                window.location.href =
                    role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN
                        ? '/admin/login'
                        : '/login';
            }
        },
    });
};

export const useChangePasswordMutation = (role: UserRoles) => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ data, reauthConfirmationToken }: { data: Record<string, string>; reauthConfirmationToken?: string }) =>
            accountApiService.changePassword(role, data, reauthConfirmationToken),
        onSuccess: () => {
            queryClient.clear();
            if (typeof document !== 'undefined') {
                document.cookie = `${COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
                document.cookie = `${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
                document.cookie = `${COOKIE_NAMES.ADMIN_ACCESS_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
                document.cookie = `${COOKIE_NAMES.ADMIN_REFRESH_TOKEN}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;`;
            }
            if (typeof window !== 'undefined') {
                window.location.href =
                    role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN
                        ? '/admin/login'
                        : '/login';
            }
        },
    });
};

export const useChangeEmailRequestMutation = (role: UserRoles) => {
    return useMutation({
        mutationFn: ({ newEmail, reauthConfirmationToken }: { newEmail: string; reauthConfirmationToken?: string }) =>
            accountApiService.changeEmailRequest(role, newEmail, reauthConfirmationToken),
    });
};

export const useChangeEmailConfirmMutation = (role: UserRoles) => {
    return useMutation({
        mutationFn: (code: string) => accountApiService.changeEmailConfirm(role, code),
    });
};

export const useDeactivateAccountMutation = (role: UserRoles) => {
    return useMutation({
        mutationFn: (reauthConfirmationToken?: string) =>
            accountApiService.deactivateAccount(role, reauthConfirmationToken),
    });
};

export const useScheduleAccountDeletionMutation = (role: UserRoles) => {
    return useMutation({
        mutationFn: (reauthConfirmationToken?: string) =>
            accountApiService.scheduleAccountDeletion(role, reauthConfirmationToken),
    });
};
