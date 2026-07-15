import { Module } from '@nestjs/common';
import { AuthTokenService } from '@/module-auth-token/services/auth-token.service';
import { JwtModule } from '@nestjs/jwt';
import { TokenService } from '@/module-auth-token/services/token.service';
import { ConfigModule, ConfigService } from '@nestjs/config';
import { getJWTConfig } from '@config/jwt.config';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { KnownDeviceService } from '@/module-auth-token/services/known-device.service';
import { ReauthConfirmationRepository } from '@/module-auth-token/repository/reauth-confirmation.repository';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { SessionService } from '@/module-auth-token/services/session.service';
import { DeviceIdService } from '@/module-auth-token/services/device-id.service';
import { AuthTokenPayloadService } from '@/module-auth-token/services/auth-token-payload.service';
import { RefreshTokenVerificationService } from '@/module-auth-token/services/refresh-token-verification.service';
import { RefreshTokenStorageService } from '@/module-auth-token/services/refresh-token-storage.service';
import { TokenCleanupService } from '@/module-auth-token/services/token-cleanup.service';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { RedisModule } from '@/common/redis/redis.module';
import { EncryptModule } from '@/module-encrypt/encrypt.module';
import { GeoIpService } from '@/module-auth-token/services/geoip.service';
import { DevelopmentGeoIpProvider } from '@/module-auth-token/services/development-geoip.provider';
import { RiskPolicyService } from '@/module-auth-token/services/risk-policy.service';
import { LoginChallengeService } from '@/module-auth-token/services/login-challenge.service';
import { UserModule } from '@/module-user/user.module';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { LoginChallengeResendPolicyService } from '@/module-auth-token/services/login-challenge-resend-policy.service';
import { MailModule } from '@/module-mail/mail.module';
import { EmailVerificationRepository } from '@/module-auth-token/repository/email-verification.repository';
import { EmailVerificationService } from '@/module-auth-token/services/email-verification.service';

@Module({
    imports: [
        ConfigModule,
        JwtModule.registerAsync({
            imports: [ConfigModule],
            inject: [ConfigService],
            useFactory: getJWTConfig,
        }),
        RedisModule,
        EncryptModule,
        UserModule,
        MailModule,
    ],
    providers: [
        AuthTokenService,
        TokenService,
        AuthTokenRepository,
        SecurityEventRepository,
        SecurityEventService,
        LoginChallengeRepository,
        KnownDeviceRepository,
        KnownDeviceService,
        ReauthConfirmationRepository,
        SessionRepository,
        SessionService,
        DeviceIdService,
        AuthTokenPayloadService,
        RefreshTokenVerificationService,
        RefreshTokenStorageService,
        TokenCleanupService,
        RedisTokenService,
        GeoIpService,
        {
            provide: 'GEOIP_PROVIDER',
            useClass: DevelopmentGeoIpProvider,
        },
        RiskPolicyService,
        LoginChallengeService,
        LoginChallengeResendPolicyService,
        ReauthConfirmationService,
        EmailVerificationRepository,
        EmailVerificationService,
    ],
    exports: [
        AuthTokenService,
        TokenService,
        AuthTokenRepository,
        SecurityEventRepository,
        SecurityEventService,
        LoginChallengeRepository,
        KnownDeviceRepository,
        KnownDeviceService,
        ReauthConfirmationRepository,
        SessionRepository,
        SessionService,
        DeviceIdService,
        RedisTokenService,
        GeoIpService,
        RiskPolicyService,
        LoginChallengeService,
        LoginChallengeResendPolicyService,
        ReauthConfirmationService,
        EmailVerificationRepository,
        EmailVerificationService,
    ],
})
export class AuthTokenModule {}
