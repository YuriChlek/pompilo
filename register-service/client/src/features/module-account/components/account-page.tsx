import { AccountCenter } from '@/features/module-account/components/account-center';

export async function AccountPage() {
    const title = 'Account security';
    const description = 'Manage your password, email, active sessions, privacy, and account lifecycle.';

    return <AccountCenter title={title} description={description} />;
}
