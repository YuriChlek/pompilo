import { Module } from '@nestjs/common';
import { JwtModule } from '@nestjs/jwt';
import { AuthModule } from '@/module-auth/auth.module';
import { UserModule } from '@/module-user/user.module';
import { IdentityOutboxController } from '@/module-integration/controllers/identity-outbox.controller';
import { TradingOnboardingController } from '@/module-integration/controllers/trading-onboarding.controller';
import { ServiceTokenGuard } from '@/module-integration/guards/service-token.guard';
import { IdentityOutboxRepository } from '@/module-integration/repository/identity-outbox.repository';
import { IdentityIntegrationService } from '@/module-integration/services/identity-integration.service';

@Module({
    imports: [AuthModule, JwtModule.register({}), UserModule],
    controllers: [IdentityOutboxController, TradingOnboardingController],
    providers: [IdentityIntegrationService, IdentityOutboxRepository, ServiceTokenGuard],
    exports: [IdentityIntegrationService, IdentityOutboxRepository],
})
export class IntegrationModule {}
