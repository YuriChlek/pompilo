import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';
import { CustomerSessionState } from '@/features/module-auth/enums/customer-session-state.enums';

export interface CookieSource {
    get: (name: string) => { value: string } | undefined | null;
}

/**
 * Pure helper that inspects customer access and refresh cookie presence.
 *
 * It returns:
 * - NONE: no active customer session tokens are present;
 * - CUSTOMER: at least one customer token (access or refresh) is present.
 *
 * Note: User identity resolution is handled via the /me endpoint.
 */
export function resolveCustomerSessionState(cookies: CookieSource): CustomerSessionState {
    const hasCustomerAccess = Boolean(cookies.get(COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN)?.value);
    const hasCustomerRefresh = Boolean(cookies.get(COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN)?.value);

    if (hasCustomerAccess || hasCustomerRefresh) {
        return CustomerSessionState.CUSTOMER;
    }

    return CustomerSessionState.NONE;
}
