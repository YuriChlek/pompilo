import { LoginForm } from '@/features/module-auth/components/auth-forms/login-form';
import { UserRoles } from '@/features/module-auth/enums/auth.enums';

export default function AdminLoginPage() {
    return <LoginForm title={'Admin Login'} mode={UserRoles.PLATFORM_ADMIN} variant="compact" />;
}
