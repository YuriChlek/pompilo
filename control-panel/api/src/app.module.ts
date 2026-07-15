import { Module } from '@nestjs/common';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { ScheduleModule } from '@nestjs/schedule';
import { UserModule } from '@/module-user/user.module';
import { AuthModule } from '@/module-auth/auth.module';
import { AuthTokenModule } from '@/module-auth-token/auth-token.module';
import { APP_GUARD, APP_INTERCEPTOR, RouterModule } from '@nestjs/core';
import { ResponseInterceptor } from '@/common/interceptors/response.interceptor';
import { EmailFlowRateLimitGuard } from '@/common/rate-limiting/guards/email-flow-rate-limit.guard';
import { EmailFlowRateLimitService } from '@/common/rate-limiting/services/email-flow-rate-limit.service';
import { AdminAuthModule } from '@/module-admin-auth/admin-auth.module';
import { BullModule } from '@nestjs/bullmq';
import { getBullMqConfig } from '@config/bull-mq.config';
import { CustomerAuthModule } from '@/module-customer-auth/customer-auth.module';
import { DrizzleModule } from '@/module-drizzle/drizzle.module';
import { AccountCoreModule } from '@/module-account/account-core.module';
import { MailModule } from '@/module-mail/mail.module';
import { API_ENV_FILE_PATHS } from '@config/api-env.config';
import { validateEnvironment } from '@config/environment.validation';
import { HealthModule } from '@/common/health/health.module';
import { LoggerModule } from '@/module-logger/logger.module';
import { IntegrationModule } from '@/module-integration/integration.module';

@Module({
    imports: [
        ConfigModule.forRoot({
            isGlobal: true,
            envFilePath: [...API_ENV_FILE_PATHS],
            validate: validateEnvironment,
        }),
        ScheduleModule.forRoot(),
        BullModule.forRootAsync({
            imports: [ConfigModule],
            inject: [ConfigService],
            useFactory: getBullMqConfig,
        }),
        AdminAuthModule,
        RouterModule.register([
            {
                path: 'admin',
                module: AdminAuthModule,
            },
        ]),
        AuthModule,
        AuthTokenModule,
        ConfigModule,
        DrizzleModule,
        UserModule,
        CustomerAuthModule,
        AccountCoreModule,
        MailModule,
        IntegrationModule,
        LoggerModule,
        HealthModule,
    ],
    providers: [
        EmailFlowRateLimitService,
        {
            provide: APP_GUARD,
            useClass: EmailFlowRateLimitGuard,
        },
        {
            provide: APP_INTERCEPTOR,
            useClass: ResponseInterceptor,
        },
    ],
    controllers: [],
})
export class AppModule {}
