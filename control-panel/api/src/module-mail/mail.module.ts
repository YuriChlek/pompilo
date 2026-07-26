import { Global, Module } from '@nestjs/common';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { MailSettingsRepository } from '@/module-mail/repository/mail-settings.repository';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { MailAuditEventRepository } from '@/module-mail/repository/mail-audit-event.repository';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import { SmtpMailService } from '@/module-mail/services/smtp-mail.service';
import { MailRedisInvalidationService } from '@/module-mail/services/mail-redis-invalidation.service';
import { BullModule } from '@nestjs/bullmq';
import { ConfigService } from '@nestjs/config';
import { getMailBootstrapConfig } from '@config/mail-bootstrap.config';
import { MailQueueEvents } from '@/module-mail/events/mail-queue-events';
import { MailOutboxRelayService } from '@/module-mail/services/mail-outbox-relay.service';
import { MailProcessorService } from '@/module-mail/services/mail-processor.service';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { MailRenderService } from '@/module-mail/services/mail-render.service';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { MailOutboxCleanupService } from '@/module-mail/services/mail-outbox-cleanup.service';
import { AdminMailController } from '@/module-mail/controllers/admin-mail.controller';
import { AdminMailTemplateController } from '@/module-mail/controllers/admin-mail-template.controller';
import { MetricsController } from '@/module-mail/controllers/metrics.controller';
import { MailMetricsService } from '@/module-mail/services/mail-metrics.service';
import { MAIL_SERVICE, MAIL_QUEUE } from '@/module-mail/constants/mail.constants';
import { RedisModule } from '@/common/redis/redis.module';
import { AdminMailActionRateLimitGuard } from '@/module-mail/guards/admin-mail-action-rate-limit.guard';
import { MailTemplatePreviewService } from '@/module-mail/services/mail-template-preview.service';

@Global()
@Module({
    imports: [
        RedisModule,
        BullModule.registerQueueAsync({
            name: MAIL_QUEUE,
            inject: [ConfigService],
            useFactory: (configService: ConfigService) => {
                const config = getMailBootstrapConfig(configService);
                return {
                    defaultJobOptions: {
                        attempts: config.retryAttempts,
                        backoff: {
                            type: config.retryBackoffType as 'exponential' | 'fixed',
                            delay: config.retryBackoffDelay,
                            jitter: config.retryBackoffJitter,
                        },
                        removeOnComplete: {
                            age: 3600, // 1 hour
                            count: 100,
                        },
                        removeOnFail: {
                            age: 3600, // 1 hour (down from 24h)
                            count: 50, // 50 jobs (down from 500)
                        },
                    },
                };
            },
        }),
    ],
    controllers: [AdminMailController, MetricsController, AdminMailTemplateController],
    providers: [
        SmtpMailService,
        {
            provide: MAIL_SERVICE,
            useExisting: SmtpMailService,
        },
        MailEncryptionService,
        MailSettingsService,
        MailReadinessService,
        MailSettingsRepository,
        MailOutboxRepository,
        MailAuditEventRepository,
        MailRedisService,
        MailRedisInvalidationService,
        MailQueueEvents,
        MailOutboxRelayService,
        MailProcessorService,
        MailRenderService,
        MailTemplateService,
        MailOutboxCleanupService,
        AdminMailActionRateLimitGuard,
        MailMetricsService,
        MailTemplatePreviewService,
    ],
    exports: [
        MAIL_SERVICE,
        MailEncryptionService,
        MailSettingsService,
        MailReadinessService,
        MailSettingsRepository,
        MailOutboxRepository,
        MailRedisService,
        MailRedisInvalidationService,
        MailRenderService,
        MailTemplateService,
    ],
})
export class MailModule {}
