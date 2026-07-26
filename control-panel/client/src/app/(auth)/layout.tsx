import { AuthLayout } from '@/features/module-auth/components/auth-layout/auth-layout';

export default function Layout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return <AuthLayout>{children}</AuthLayout>;
}
