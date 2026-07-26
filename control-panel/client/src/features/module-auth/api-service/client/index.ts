import { apiClient } from '@/lib/http-client/http-client';
import type { HttpResponse } from '@/lib/http-client/interfaces/http-client.interfaces';
import type {
    AuthApi,
    User,
    CheckpointResponse,
    ReauthResponse,
} from '@/features/module-auth/interfaces/auth.interfaces';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import type { LoginData, RegisterData } from '@/features/module-auth/types/auth.types';

function isAdminRole(role?: UserRoles): boolean {
    return role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN;
}

const getAccountPrefix = (role: UserRoles) => {
    if (isAdminRole(role)) {
        return '/admin';
    }
    return '/account';
};

export const authService: AuthApi = {
    async login(login: string, password: string, role?: UserRoles): Promise<User | CheckpointResponse | null> {
        const path = isAdminRole(role) ? '/admin/login' : '/login';

        const requestData: LoginData = {
            login,
            password,
        };

        const response: HttpResponse<User | CheckpointResponse> = await apiClient.post<User | CheckpointResponse, LoginData>(
            path,
            requestData,
        );

        if (!response.success) {
            throw new Error(response.message);
        }

        return response.data as unknown as User | CheckpointResponse;
    },
    async register(data: RegisterData): Promise<User> {
        try {
            const path = '/auth/register';

            const response: HttpResponse<User> = await apiClient.post<User, RegisterData>(
                path,
                data,
            );

            if (!response.success) {
                throw new Error(response.message);
            }

            return response.data as unknown as User;
        } catch (error) {
            throw error;
        }
    },
    async logout(role?: UserRoles): Promise<boolean> {
        const path = isAdminRole(role) ? '/admin/logout' : '/logout';
        const response: HttpResponse<unknown> = await apiClient.post(path, {});

        return response.success;
    },

    async getMe(role?: UserRoles): Promise<User | null> {
        try {
            const path = isAdminRole(role) ? '/admin/me' : '/me';
            const response: HttpResponse<User> = await apiClient.post(path, {});
            if (!response.success) {
                throw new Error(response.message);
            }

            return response.data as unknown as User;
        } catch (error) {
            throw error;
        }
    },

    async verifyCheckpoint(
        checkpointToken: string,
        code: string,
        role?: UserRoles,
    ): Promise<User | null> {
        const path = isAdminRole(role) ? '/admin/checkpoint/verify' : '/checkpoint/verify';

        const response: HttpResponse<User> = await apiClient.post<User, { checkpointToken: string; code: string }>(
            path,
            { checkpointToken, code },
        );

        if (!response.success) {
            throw new Error(response.message);
        }

        return response.data as unknown as User;
    },

    async resendCheckpoint(checkpointToken: string, role?: UserRoles): Promise<CheckpointResponse> {
        const path = isAdminRole(role) ? '/admin/checkpoint/resend' : '/checkpoint/resend';

        const response: HttpResponse<CheckpointResponse> = await apiClient.post<
            CheckpointResponse,
            { checkpointToken: string }
        >(path, { checkpointToken });

        if (!response.success) {
            throw new Error(response.message);
        }

        return response.data as unknown as CheckpointResponse;
    },

    async reauth(password: string, actionScope: string, role: UserRoles): Promise<ReauthResponse> {
        const path = `${getAccountPrefix(role)}/re-auth`;

        const response = await apiClient.post<ReauthResponse, { password: string; actionScope: string }>(
            path,
            { password, actionScope },
        );

        if (!response.success) {
            throw new Error(response.message);
        }

        return response.data as unknown as ReauthResponse;
    },
    async verifyEmail(token: string): Promise<boolean> {
        const path = '/verify-email';
        const response = await apiClient.post<boolean, { token: string }>(path, { token });
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data ?? false;
    },
    async resendVerification(): Promise<void> {
        const path = '/resend-verification';
        const response = await apiClient.post<void, unknown>(path, {});
        if (!response.success) {
            throw new Error(response.message);
        }
    },
    async forgotPassword(email: string): Promise<void> {
        const path = '/auth/password/forgot';
        const response = await apiClient.post<void, { email: string }>(path, { email });
        if (!response.success) {
            throw new Error(response.message);
        }
    },
    async resetPassword(token: string, newPassword: string): Promise<void> {
        const path = '/auth/password/reset';
        const response = await apiClient.post<void, { token: string; newPassword: string }>(
            path,
            { token, newPassword },
        );
        if (!response.success) {
            throw new Error(response.message);
        }
    },
};
