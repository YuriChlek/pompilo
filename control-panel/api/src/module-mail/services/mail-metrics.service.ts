import { Injectable } from '@nestjs/common';
import { InjectQueue } from '@nestjs/bullmq';
import { Queue } from 'bullmq';
import { MAIL_QUEUE } from '@/module-mail/constants/mail.constants';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';

@Injectable()
export class MailMetricsService {
    constructor(
        private readonly mailOutboxRepository: MailOutboxRepository,
        private readonly mailRedisService: MailRedisService,
        @InjectQueue(MAIL_QUEUE) private readonly mailQueue: Queue,
    ) {}

    async getMetricsAsText(): Promise<string> {
        // 1. Outbox lag
        const lag = await this.mailOutboxRepository.getOldestLagSeconds();

        // 2. Queue depth
        let queueDepth = 0;
        try {
            const counts = await this.mailQueue.getJobCounts('waiting', 'delayed', 'paused');
            queueDepth = (counts.waiting || 0) + (counts.delayed || 0) + (counts.paused || 0);
        } catch {
            // Fallback if Redis/Bull is down
        }

        // 3. Success rate and error counts
        const successCount = await this.mailRedisService.getSuccessCount();
        const errorCounts = await this.mailRedisService.getErrorCounts();
        const totalAttempts =
            successCount +
            errorCounts.network +
            errorCounts.auth +
            errorCounts['rate-limit'] +
            errorCounts.rejection;
        const successRate = totalAttempts > 0 ? successCount / totalAttempts : 1.0;

        // 4. Format Prometheus metrics text
        const lines = [
            '# HELP mail_outbox_lag_seconds Lag in seconds of the oldest pending/failed outbox message',
            '# TYPE mail_outbox_lag_seconds gauge',
            `mail_outbox_lag_seconds ${lag}`,
            '',
            '# HELP bullmq_queue_depth Count of waiting, delayed, and paused jobs in the email queue',
            '# TYPE bullmq_queue_depth gauge',
            `bullmq_queue_depth ${queueDepth}`,
            '',
            '# HELP smtp_delivery_success_rate Successful delivery count divided by total delivery attempts',
            '# TYPE smtp_delivery_success_rate gauge',
            `smtp_delivery_success_rate ${successRate.toFixed(4)}`,
            '',
            '# HELP smtp_delivery_errors_total Total number of SMTP delivery errors categorized by failure type',
            '# TYPE smtp_delivery_errors_total counter',
            `smtp_delivery_errors_total{category="network"} ${errorCounts.network}`,
            `smtp_delivery_errors_total{category="auth"} ${errorCounts.auth}`,
            `smtp_delivery_errors_total{category="rate-limit"} ${errorCounts['rate-limit']}`,
            `smtp_delivery_errors_total{category="rejection"} ${errorCounts.rejection}`,
        ];

        return lines.join('\n') + '\n';
    }
}
