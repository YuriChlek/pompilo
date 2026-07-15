export enum UserRoles {
    USER = 'user',
    PLATFORM_ADMIN = 'platformAdmin',
    SUPER_ADMIN = 'superAdmin',
}

export enum COOKIE_NAMES {
    CUSTOMER_ACCESS_TOKEN = 'customerAccessToken',
    CUSTOMER_REFRESH_TOKEN = 'customerRefreshToken',
    ADMIN_ACCESS_TOKEN = 'adminAccessToken',
    ADMIN_REFRESH_TOKEN = 'adminRefreshToken',
    DEVICE_ID = 'deviceId',
}

export type AuthRealm = 'customer' | 'admin';

export function deriveAuthRealmFromRole(role: UserRoles): AuthRealm {
    if (role === UserRoles.PLATFORM_ADMIN || role === UserRoles.SUPER_ADMIN) {
        return 'admin';
    }
    return 'customer';
}
