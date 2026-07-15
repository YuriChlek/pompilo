import { MailSettingsPage } from '@/features/module-admin-mail/components/mail-settings-page';
import { Metadata } from 'next';

export const metadata: Metadata = {
    title: 'Mail Settings | Admin Panel',
};

export default function Page() {
    return <MailSettingsPage />;
}
