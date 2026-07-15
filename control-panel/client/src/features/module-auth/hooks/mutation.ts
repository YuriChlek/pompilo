import { useMutation, useQueryClient } from '@tanstack/react-query';
import { authService } from '@/features/module-auth/api-service/client';
import { getUserQueryKey } from '@/features/module-auth/hooks/query';
import { useRouter } from 'next/navigation';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';
import { CUSTOMER_DEFAULT_MENU_ITEM } from '@/features/module-menu/config/menu.config';
import type { CheckpointResponse, User } from '@/features/module-auth/interfaces/auth.interfaces';
import type { LoginData, RegisterData } from '@/features/module-auth/types/auth.types';

const isAdminRole = (role?: UserRoles) =>
    role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN;

const ADMIN_DEFAULT_PATH = '/admin/dashboard';

const getPostAuthRedirectPath = (role: UserRoles): string =>
    isAdminRole(role) ? ADMIN_DEFAULT_PATH : CUSTOMER_DEFAULT_MENU_ITEM.href;

export const useRegister = (shouldRedirect = true) => {
    const queryClient = useQueryClient();
    const router = useRouter();

    return useMutation({
        mutationFn: async (data: RegisterData) => {
            return await authService.register(data);
        },

        onSuccess: newUser => {
            if (newUser) {
                queryClient.setQueryData(getUserQueryKey(newUser.role), newUser);

                if (shouldRedirect) {
                    router.push('/verify-email');
                }
            }
        },

        onError: error => {
            console.log(error.message);
        },
    });
};

export const useLogin = () => {
    const queryClient = useQueryClient();
    const router = useRouter();

    return useMutation({
        mutationFn: async (loginData: LoginData) => {
            const { login, password, role } = loginData;

            return await authService.login(login, password, role);
        },

        onSuccess: result => {
            if (result) {
                if ('checkpointRequired' in result) {
                    // Do NOT perform role redirect. The login component or page will handle the checkpoint UI transition.
                    return;
                }
                const user = result as User;
                const { role } = user;
                queryClient.setQueryData(getUserQueryKey(role), user);

                router.push(getPostAuthRedirectPath(role));
            }
        },

        onError: error => {
            console.log(error.message);
        },
    });
};

export const useLogout = () => {
    const queryClient = useQueryClient();
    const router = useRouter();

    return useMutation({
        mutationFn: async (role?: UserRoles) => {
            return await authService.logout(role);
        },

        onSuccess: (_result, role) => {
            queryClient.clear();
            router.refresh();
            router.replace(isAdminRole(role) ? '/admin/login' : '/login');
        },

        onError: error => {
            console.log(error.message);
        },
    });
};

export const useVerifyCheckpoint = () => {
    const queryClient = useQueryClient();
    const router = useRouter();

    return useMutation({
        mutationFn: async ({
            checkpointToken,
            code,
            role,
        }: {
            checkpointToken: string;
            code: string;
            role?: UserRoles;
        }) => {
            return await authService.verifyCheckpoint(checkpointToken, code, role);
        },

        onSuccess: newUser => {
            if (newUser) {
                const { role } = newUser;
                queryClient.setQueryData(getUserQueryKey(role), newUser);

                router.push(getPostAuthRedirectPath(role));
            }
        },

        onError: error => {
            console.log(error.message);
        },
    });
};

export const useResendCheckpoint = () => {
    return useMutation({
        mutationFn: async ({
            checkpointToken,
            role,
        }: {
            checkpointToken: string;
            role?: UserRoles;
        }): Promise<CheckpointResponse> => {
            return await authService.resendCheckpoint(checkpointToken, role);
        },

        onError: error => {
            console.log(error.message);
        },
    });
};

export const useVerifyEmail = () => {
    return useMutation({
        mutationFn: async (token: string) => {
            return await authService.verifyEmail(token);
        },
        onError: error => {
            console.log(error.message);
        },
    });
};

export const useResendVerification = () => {
    return useMutation({
        mutationFn: async () => {
            return await authService.resendVerification();
        },
        onError: error => {
            console.log(error.message);
        },
    });
};

export const useForgotPassword = () => {
    return useMutation({
        mutationFn: async (email: string) => {
            return await authService.forgotPassword(email);
        },
        onError: error => {
            console.log(error.message);
        },
    });
};

export const useResetPassword = () => {
    const router = useRouter();

    return useMutation({
        mutationFn: async ({ token, newPassword }: { token: string; newPassword: string }) => {
            return await authService.resetPassword(token, newPassword);
        },
        onSuccess: () => {
            router.replace('/login');
        },
        onError: error => {
            console.log(error.message);
        },
    });
};
