import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';

@Injectable()
export class TokenCleanupService {
    private readonly logger = new Logger(TokenCleanupService.name);

    constructor(
        private readonly authTokenRepository: AuthTokenRepository,
        private readonly reauthConfirmationService: ReauthConfirmationService,
        private readonly sessionRepository: SessionRepository,
        private readonly knownDeviceRepository: KnownDeviceRepository,
        private readonly loginChallengeRepository: LoginChallengeRepository,
        private readonly securityEventService: SecurityEventService,
    ) {}

    @Cron(CronExpression.EVERY_DAY_AT_MIDNIGHT, { name: 'token-cleanup-job' })
    async handleCron() {
        this.logger.log('Starting scheduled cleanup of expired refresh tokens');
        try {
            const deletedCount = await this.authTokenRepository.deleteExpiredTokens();
            this.logger.log(`Successfully deleted ${deletedCount} expired refresh tokens`);
        } catch (error) {
            this.logger.error('Failed to cleanup expired refresh tokens', error);
        }

        this.logger.log('Starting scheduled cleanup of expired reauth confirmations');
        try {
            // Retention period: 24 hours
            const retentionPeriod = 24 * 60 * 60 * 1000;
            const deletedCount =
                await this.reauthConfirmationService.expireReauthConfirmations(retentionPeriod);
            this.logger.log(`Successfully deleted ${deletedCount} expired reauth confirmations`);
        } catch (error) {
            this.logger.error('Failed to cleanup expired reauth confirmations', error);
        }

        this.logger.log('Starting scheduled cleanup of expired login challenges');
        try {
            // Retention period: 24 hours
            const retentionPeriod = 24 * 60 * 60 * 1000;
            const deletedCount =
                await this.loginChallengeRepository.deleteExpiredChallenges(retentionPeriod);
            this.logger.log(`Successfully deleted ${deletedCount} expired login challenges`);
        } catch (error) {
            this.logger.error('Failed to cleanup expired login challenges', error);
        }

        this.logger.log('Starting scheduled cleanup of expired sessions');
        try {
            // Retention period: 30 days
            const retentionPeriod = 30 * 24 * 60 * 60 * 1000;
            const deletedCount =
                await this.sessionRepository.deleteExpiredSessions(retentionPeriod);
            this.logger.log(`Successfully deleted ${deletedCount} expired sessions`);
        } catch (error) {
            this.logger.error('Failed to cleanup expired sessions', error);
        }

        this.logger.log('Starting scheduled cleanup of expired known devices');
        try {
            // Retention period: 30 days
            const retentionPeriod = 30 * 24 * 60 * 60 * 1000;
            const deletedCount =
                await this.knownDeviceRepository.deleteExpiredKnownDevices(retentionPeriod);
            this.logger.log(`Successfully deleted ${deletedCount} expired known devices`);
        } catch (error) {
            this.logger.error('Failed to cleanup expired known devices', error);
        }

        this.logger.log('Starting scheduled cleanup of expired security events');
        try {
            // Retention period: 90 days
            const retentionPeriod = 90 * 24 * 60 * 60 * 1000;
            const deletedCount =
                await this.securityEventService.deleteExpiredEvents(retentionPeriod);
            this.logger.log(`Successfully deleted ${deletedCount} expired security events`);
        } catch (error) {
            this.logger.error('Failed to cleanup expired security events', error);
        }
    }
}
