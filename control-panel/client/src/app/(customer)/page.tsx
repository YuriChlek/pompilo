import { redirect } from 'next/navigation';
import { CUSTOMER_DEFAULT_MENU_ITEM } from '@/features/module-menu/config/menu.config';

export default async function HomePage() {
    redirect(CUSTOMER_DEFAULT_MENU_ITEM.href);
}
