import type { UserRoles } from '@/features/module-auth/enums/auth.enums';

export interface LogoutButtonProps {
    isAuthenticated: boolean;
    role?: UserRoles;
    loginPath?: string;
    className?: string;
    forceCompact?: boolean;
}

export interface LoginFormProps {
    mode?: UserRoles;
    title?: string;
    variant?: 'modern' | 'compact';
}

export interface UserBadgeProps {
    userName: string;
    href?: string;
}
