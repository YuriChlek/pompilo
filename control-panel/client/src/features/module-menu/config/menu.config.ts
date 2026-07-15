import { MenuTypes, MenuPlacement } from '@/features/module-menu/enums/menu.enums';
import type { MenuItem } from '@/features/module-menu/interfaces/menu.interfaces';
import {
    faHouse,
    faKey,
    faShieldHalved,
} from '@fortawesome/free-solid-svg-icons';

export const CUSTOMER_HOME_MENU_ITEM: MenuItem = {
    title: 'Home',
    href: '/',
    icon: faHouse,
    placement: MenuPlacement.PRIMARY,
    mobile: true,
};

export const CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM: MenuItem = {
    title: 'Account Security',
    href: '/account/security',
    icon: faShieldHalved,
    placement: MenuPlacement.PRIMARY,
    mobile: true,
    matchPaths: ['/account/security'],
};

export const CUSTOMER_API_KEYS_MENU_ITEM: MenuItem = {
    title: 'API Keys',
    href: '/account/api-keys',
    icon: faKey,
    placement: MenuPlacement.PRIMARY,
    mobile: true,
    matchPaths: ['/account/api-keys'],
};

export const CUSTOMER_DEFAULT_MENU_ITEM = CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM;

export const identityCustomerMenu: Array<MenuItem> = [
    CUSTOMER_HOME_MENU_ITEM,
    CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM,
    CUSTOMER_API_KEYS_MENU_ITEM,
];

const guestMenu: Array<MenuItem> = [];

export const getMenu = (menuType: MenuTypes): Array<MenuItem> => {
    const config: Record<MenuTypes, Array<MenuItem>> = {
        [MenuTypes.CUSTOMER]: identityCustomerMenu,
        [MenuTypes.GUEST]: guestMenu,
    };
    return config[menuType];
};
