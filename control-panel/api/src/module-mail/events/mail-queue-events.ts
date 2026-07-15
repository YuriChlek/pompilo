import { Injectable, Logger } from '@nestjs/common';
import { InjectQueue, OnQueueEvent, QueueEventsHost, QueueEventsListener } from '@nestjs/bullmq';
import { Queue } from 'bullmq';
import { MAIL_POISON_FAILURE_CODES, MAIL_QUEUE } from '@/module-mail/constants/mail.constants';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';

@Injectable()
@QueueEventsListener(MAIL_QUEUE)
export class MailQueueEvents extends QueueEventsHost {
    private readonly logger = new Logger(MailQueueEvents.name);

    constructor(
        private readonly mailReadinessService: MailReadinessService,
        private readonly mailOutboxRepository: MailOutboxRepository,
        @InjectQueue(MAIL_QUEUE) private readonly mailQueue: Queue,
    ) {
        super();
    }

    @OnQueueEvent('failed')
    async onFailed({
        jobId,
        failedReason,
    }: {
        jobId: string;
        failedReason: string;
        prev?: string;
    }) {
        const isUnrecoverable = Object.values(MAIL_POISON_FAILURE_CODES).some(code =>
            failedReason.startsWith(code),
        );

        this.logger.error(`Mail job ${jobId} failed. Reason: ${failedReason}`);

        try {
            const job = await this.mailQueue.getJob(jobId);
            if (job) {
                const maxAttempts = job.opts.attempts || 1;
                if (job.attemptsMade < maxAttempts && !isUnrecoverable) {
                    this.logger.warn(
                        `Mail job ${jobId} failed attempt ${job.attemptsMade}/${maxAttempts}. Will retry.`,
                    );
                    return;
                }
            }
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            this.logger.error(`Failed to retrieve job ${jobId} from queue: ${message}`);
        }

        // Mark outbox record as failed
        await this.mailOutboxRepository.markAsFailed(jobId, failedReason);
    }

    @OnQueueEvent('completed')
    async onCompleted({ jobId }: { jobId: string; returnvalue: string; prev?: string }) {
        this.logger.log(`Mail job ${jobId} completed successfully.`);

        // Phase 3: Signal healthy state on successful completion to allow faster recovery
        await this.mailReadinessService.setReadinessStatus('healthy', undefined, 3600);
    }
}
