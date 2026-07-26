import { Module } from '@nestjs/common';
import { AuthService } from './services/auth.service';
import { UserModule } from '@/module-user/user.module';
import { AuthTokenModule } from '@/module-auth-token/auth-token.module';
import { JwtCustomerAuthStrategy } from '@/module-auth/strategies/jwt-customer-auth.strategy';
import { JwtAdminAuthStrategy } from '@/module-auth/strategies/jwt-admin-auth.strategy';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { JwtAuthGuard } from '@/module-auth/guards/jwt-auth.guard';
import { RolesGuard } from '@/module-auth/guards/roles.guard';
import { EmailVerifiedGuard } from '@/module-auth/guards/email-verified.guard';

@Module({
    imports: [UserModule, AuthTokenModule],
    providers: [
        AuthService,
        AuthSessionService,
        JwtCustomerAuthStrategy,
        JwtAdminAuthStrategy,
        JwtAuthGuard,
        RolesGuard,
        EmailVerifiedGuard,
    ],
    exports: [AuthService, JwtAuthGuard, RolesGuard, EmailVerifiedGuard, AuthSessionService],
})
export class AuthModule {}
