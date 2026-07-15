import { LoginForm } from '@/features/module-auth/components/auth-forms/login-form';
import { AuthLayout } from '@/features/module-auth/components/auth-layout/auth-layout';

export default function LoginPage() {
    return (
        <AuthLayout>
            <LoginForm />
        </AuthLayout>
    );
}
