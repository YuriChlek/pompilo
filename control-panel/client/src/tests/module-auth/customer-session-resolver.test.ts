import { describe, it, expect } from 'vitest';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';
import { CustomerSessionState } from '@/features/module-auth/enums/customer-session-state.enums';
import {
    resolveCustomerSessionState,
    type CookieSource,
} from '@/features/module-auth/server/customer-session-resolver';

describe('resolveCustomerSessionState', () => {
    const createCookies = (cookieMap: Record<string, string>): CookieSource => ({
        get: (name: string) => {
            const value = cookieMap[name];
            return value ? { value } : undefined;
        },
    });

    it('should return NONE when no customer cookies are present', () => {
        const cookies = createCookies({});
        expect(resolveCustomerSessionState(cookies)).toBe(CustomerSessionState.NONE);
    });

    it('should return NONE when only admin cookies are present', () => {
        const cookies = createCookies({
            [COOKIE_NAMES.ADMIN_ACCESS_TOKEN]: 'token',
        });
        expect(resolveCustomerSessionState(cookies)).toBe(CustomerSessionState.NONE);
    });

    it('should return CUSTOMER when customer access token is present', () => {
        const cookies = createCookies({
            [COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN]: 'token',
        });
        expect(resolveCustomerSessionState(cookies)).toBe(CustomerSessionState.CUSTOMER);
    });

    it('should return CUSTOMER when customer refresh token is present', () => {
        const cookies = createCookies({
            [COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN]: 'token',
        });
        expect(resolveCustomerSessionState(cookies)).toBe(CustomerSessionState.CUSTOMER);
    });

    it('should ignore admin cookies even if customer tokens are present', () => {
        const cookies = createCookies({
            [COOKIE_NAMES.ADMIN_ACCESS_TOKEN]: 'admin',
            [COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN]: 'customer',
        });
        expect(resolveCustomerSessionState(cookies)).toBe(CustomerSessionState.CUSTOMER);
    });
});
