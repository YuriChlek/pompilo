import { EmailVerification } from '@/features/module-auth/components/email-verification/email-verification';
import { AuthLayout } from '@/features/module-auth/components/auth-layout/auth-layout';

export default function VerifyEmailPage() {
    return (
        <AuthLayout>
            <EmailVerification />
        </AuthLayout>
    );
}
