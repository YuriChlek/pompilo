import { ResetPasswordForm } from '@/features/module-auth/components/password-recovery/reset-password-form';
import { AuthLayout } from '@/features/module-auth/components/auth-layout/auth-layout';

export default function AuthResetPasswordPage() {
    return (
        <AuthLayout>
            <ResetPasswordForm />
        </AuthLayout>
    );
}
