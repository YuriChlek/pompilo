import type { User } from '@/features/module-auth/interfaces/auth.interfaces';

export type UserResponse = {
    success?: boolean;
    data?: User;
};
