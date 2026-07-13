import { apiClient } from '@/lib/http-client/http-client';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { UserSession } from '@/features/module-account/interfaces/account.interfaces';

export const accountApiService = {
    async getActiveSessions(_role: UserRoles): Promise<UserSession[]> {
        void _role;
        const response = await apiClient.get<UserSession[]>('/account/sessions');
        if (!response.success) {
            throw new Error(response.message || 'Failed to retrieve active sessions');
        }
        return response.data as unknown as UserSession[];
    },

    async revokeSession(_role: UserRoles, sessionId: string): Promise<void> {
        void _role;
        const response = await apiClient.delete<void>(`/account/sessions/${sessionId}`, { retry: false });
        if (!response.success) {
            throw new Error(response.message || 'Failed to revoke session');
        }
    },

    async revokeOtherSessions(_role: UserRoles, reauthConfirmationToken?: string): Promise<void> {
        void _role;
        const response = reauthConfirmationToken
            ? await apiClient.delete<void>('/account/sessions/others', {
                  retry: false,
                  headers: { 'X-Reauth-Confirmation': reauthConfirmationToken },
              })
            : await apiClient.delete<void>('/account/sessions/others', { retry: false });

        if (!response.success) {
            throw new Error(response.message || 'Failed to revoke other sessions');
        }
    },

    async logoutAllSessions(_role: UserRoles): Promise<void> {
        void _role;
        const response = await apiClient.delete<void>('/account/sessions', { retry: false });
        if (!response.success) {
            throw new Error(response.message || 'Failed to logout all sessions');
        }
    },

    async changePassword(_role: UserRoles, data: Record<string, string>, reauthConfirmationToken?: string): Promise<void> {
        void _role;
        const response = reauthConfirmationToken
            ? await apiClient.post<void, typeof data>(
                  '/account/password/change',
                  data,
                  { headers: { 'X-Reauth-Confirmation': reauthConfirmationToken } }
              )
            : await apiClient.post<void, typeof data>('/account/password/change', data);

        if (!response.success) {
            throw new Error(response.message || 'Failed to change password');
        }
    },

    async changeEmailRequest(_role: UserRoles, newEmail: string, reauthConfirmationToken?: string): Promise<void> {
        void _role;
        const data = { newEmail };
        const response = reauthConfirmationToken
            ? await apiClient.post<void, typeof data>(
                  '/account/email/change/request',
                  data,
                  { headers: { 'X-Reauth-Confirmation': reauthConfirmationToken } }
              )
            : await apiClient.post<void, typeof data>('/account/email/change/request', data);

        if (!response.success) {
            throw new Error(response.message || 'Failed to request email change');
        }
    },

    async changeEmailConfirm(_role: UserRoles, code: string): Promise<void> {
        void _role;
        const data = { code };
        const response = await apiClient.post<void, typeof data>('/account/email/change/confirm', data);
        if (!response.success) {
            throw new Error(response.message || 'Failed to confirm email change');
        }
    },

    async deactivateAccount(_role: UserRoles, reauthConfirmationToken?: string): Promise<void> {
        void _role;
        const response = reauthConfirmationToken
            ? await apiClient.post<void, undefined>(
                  '/account/deactivate',
                  undefined,
                  { headers: { 'X-Reauth-Confirmation': reauthConfirmationToken } }
              )
            : await apiClient.post<void, undefined>('/account/deactivate', undefined);

        if (!response.success) {
            throw new Error(response.message || 'Failed to deactivate account');
        }
    },

    async scheduleAccountDeletion(_role: UserRoles, reauthConfirmationToken?: string): Promise<void> {
        void _role;
        const response = reauthConfirmationToken
            ? await apiClient.delete<void>('/account', {
                  retry: false,
                  headers: { 'X-Reauth-Confirmation': reauthConfirmationToken },
              })
            : await apiClient.delete<void>('/account', { retry: false });

        if (!response.success) {
            throw new Error(response.message || 'Failed to schedule account deletion');
        }
    },
};
