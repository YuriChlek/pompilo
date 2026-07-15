import { ForgotPasswordForm } from '@/features/module-auth/components/password-recovery/forgot-password-form';
import { AuthLayout } from '@/features/module-auth/components/auth-layout/auth-layout';

export default function AuthForgotPasswordPage() {
    return (
        <AuthLayout>
            <ForgotPasswordForm />
        </AuthLayout>
    );
}
