import { Module } from '@nestjs/common';
import { AuthModule } from '@/module-auth/auth.module';
import { CustomerAuthService } from '@/module-customer-auth/services/customer-auth.service';
import { CustomerAuthController } from '@/module-customer-auth/controllers/customer-auth.controller';
import { DrizzleModule } from '@/module-drizzle/drizzle.module';
import { UserModule } from '@/module-user/user.module';
import { AuthTokenModule } from '@/module-auth-token/auth-token.module';
import { IntegrationModule } from '@/module-integration/integration.module';

@Module({
    imports: [AuthModule, DrizzleModule, UserModule, AuthTokenModule, IntegrationModule],
    controllers: [CustomerAuthController],
    providers: [CustomerAuthService],
    exports: [CustomerAuthService],
})
export class CustomerAuthModule {}
