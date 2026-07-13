import type { MenuTypes, MenuPlacement } from '@/features/module-menu/enums/menu.enums';
import type { IconDefinition } from '@fortawesome/fontawesome-svg-core';

export interface MenuItem {
    title: string;
    href: string;
    icon?: string | IconDefinition;
    placement?: MenuPlacement;
    disabled?: boolean;
    matchPaths?: string[];
    mobile?: boolean;
}

export interface MenuProps {
    menuType: MenuTypes;
}
