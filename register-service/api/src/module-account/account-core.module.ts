import { Module } from '@nestjs/common';
import { UserSettingsRepository } from '@/module-account/repository/user-settings.repository';
import { PasswordResetChallengeRepository } from '@/module-account/repository/password-reset-challenge.repository';
import { EmailChangeChallengeRepository } from '@/module-account/repository/email-change-challenge.repository';
import { AccountSecurityRepository } from '@/module-account/repository/account-security.repository';
import { AccountSettingsService } from '@/module-account/services/account-settings.service';
import { AccountCleanupService } from '@/module-account/services/account-cleanup.service';
import { UserModule } from '@/module-user/user.module';
import { AuthTokenModule } from '@/module-auth-token/auth-token.module';
import { MailModule } from '@/module-mail/mail.module';
import { AccountSecurityController } from '@/module-account/controllers/account-security.controller';
import { AuthModule } from '@/module-auth/auth.module';
import { IntegrationModule } from '@/module-integration/integration.module';

@Module({
    imports: [UserModule, AuthTokenModule, AuthModule, MailModule, IntegrationModule],
    controllers: [AccountSecurityController],
    providers: [
        UserSettingsRepository,
        PasswordResetChallengeRepository,
        EmailChangeChallengeRepository,
        AccountSecurityRepository,
        AccountSettingsService,
        AccountCleanupService,
    ],
    exports: [
        UserSettingsRepository,
        PasswordResetChallengeRepository,
        EmailChangeChallengeRepository,
        AccountSecurityRepository,
        AccountSettingsService,
    ],
})
export class AccountCoreModule {}
