import { Injectable, Logger } from '@nestjs/common';
import { Cron } from '@nestjs/schedule';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';

@Injectable()
export class MailOutboxCleanupService {
    private readonly logger = new Logger(MailOutboxCleanupService.name);

    constructor(private readonly outboxRepository: MailOutboxRepository) {}

    // Runs daily at 3:00 AM (off-peak hours)
    @Cron('0 3 * * *')
    async cleanup() {
        this.logger.log('Starting scheduled cleanup of sent and failed mail outbox records');
        try {
            // sent records retention: 24 hours
            const sentOlderThan = new Date();
            sentOlderThan.setHours(sentOlderThan.getHours() - 24);

            // failed records retention: 7 days
            const failedOlderThan = new Date();
            failedOlderThan.setDate(failedOlderThan.getDate() - 7);

            const sentDeleted = await this.outboxRepository.deleteOldRecordsByStatusInBatches(
                'sent',
                sentOlderThan,
            );

            const failedDeleted = await this.outboxRepository.deleteOldRecordsByStatusInBatches(
                'failed',
                failedOlderThan,
            );

            if (sentDeleted > 0 || failedDeleted > 0) {
                this.logger.log(
                    `Cleaned up ${sentDeleted} sent records and ${failedDeleted} failed outbox records.`,
                );
            } else {
                this.logger.log('No outbox records required cleanup.');
            }
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Failed to cleanup mail outbox: ${message}`);
        }
    }
}
