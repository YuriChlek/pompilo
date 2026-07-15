import { describe, expect, it } from 'vitest';
import {
    CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM,
    CUSTOMER_API_KEYS_MENU_ITEM,
    CUSTOMER_HOME_MENU_ITEM,
    getMenu,
} from '@/features/module-menu/config/menu.config';
import { MenuTypes } from '@/features/module-menu/enums/menu.enums';

describe('customer menu config', () => {
    it('keeps only active identity menu entries', () => {
        const menu = getMenu(MenuTypes.CUSTOMER);

        expect(menu).toEqual([
            CUSTOMER_HOME_MENU_ITEM,
            CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM,
            CUSTOMER_API_KEYS_MENU_ITEM,
        ]);

        expect(CUSTOMER_ACCOUNT_SECURITY_MENU_ITEM.disabled).toBeUndefined();
        expect(CUSTOMER_API_KEYS_MENU_ITEM.disabled).toBeUndefined();
        expect(menu.some(item => item.disabled)).toBe(false);
    });
});
