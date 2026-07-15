import { Injectable, Logger } from '@nestjs/common';
import { Cron, CronExpression } from '@nestjs/schedule';
import { PasswordResetChallengeRepository } from '@/module-account/repository/password-reset-challenge.repository';
import { EmailChangeChallengeRepository } from '@/module-account/repository/email-change-challenge.repository';

@Injectable()
export class AccountCleanupService {
    private readonly logger = new Logger(AccountCleanupService.name);

    constructor(
        private readonly passwordResetChallengeRepository: PasswordResetChallengeRepository,
        private readonly emailChangeChallengeRepository: EmailChangeChallengeRepository,
    ) {}

    @Cron(CronExpression.EVERY_HOUR)
    async cleanupChallenges() {
        try {
            // Retention period for used/invalidated challenges: 24 hours
            const olderThan = new Date();
            olderThan.setHours(olderThan.getHours() - 24);

            const prDeleted = await this.passwordResetChallengeRepository.cleanup(olderThan);
            const ecDeleted = await this.emailChangeChallengeRepository.cleanup(olderThan);

            if (prDeleted > 0 || ecDeleted > 0) {
                this.logger.log(
                    `Cleaned up ${prDeleted} password reset and ${ecDeleted} email change challenges.`,
                );
            }
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Failed to cleanup challenges: ${message}`);
        }
    }
}
