import type { Metadata } from 'next';
import { AdminBotConfigPage } from '@/features/module-admin-bots';

export const metadata: Metadata = {
    title: 'Bot Configuration | Admin Panel',
};

export default function Page() {
    return <AdminBotConfigPage />;
}
