import type { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';

export type AuthScope = 'admin' | 'customer';

export type AuthRealm = 'customer' | 'admin';

export type AuthScopeConfig = {
    accessCookie: COOKIE_NAMES;
    refreshCookie: COOKIE_NAMES;
    loginPath: string;
    dashboardPath: string;
    refreshPath: string;
};
