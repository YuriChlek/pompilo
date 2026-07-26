import { Test, TestingModule } from '@nestjs/testing';
import { MailMetricsService } from '@/module-mail/services/mail-metrics.service';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import { getQueueToken } from '@nestjs/bullmq';
import { MAIL_QUEUE } from '@/module-mail/constants/mail.constants';

describe('MailMetricsService', () => {
    let service: MailMetricsService;
    let outboxRepository: {
        getOldestLagSeconds: jest.Mock;
    };
    let redisService: {
        getSuccessCount: jest.Mock;
        getErrorCounts: jest.Mock;
    };
    let mockQueue: {
        getJobCounts: jest.Mock;
    };

    beforeEach(async () => {
        outboxRepository = {
            getOldestLagSeconds: jest.fn(),
        };
        redisService = {
            getSuccessCount: jest.fn(),
            getErrorCounts: jest.fn(),
        };
        mockQueue = {
            getJobCounts: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailMetricsService,
                { provide: MailOutboxRepository, useValue: outboxRepository },
                { provide: MailRedisService, useValue: redisService },
                { provide: getQueueToken(MAIL_QUEUE), useValue: mockQueue },
            ],
        }).compile();

        service = module.get<MailMetricsService>(MailMetricsService);
    });

    it('should generate Prometheus formatted metrics correctly', async () => {
        outboxRepository.getOldestLagSeconds.mockResolvedValue(45);
        mockQueue.getJobCounts.mockResolvedValue({
            waiting: 5,
            delayed: 2,
            paused: 3,
        });
        redisService.getSuccessCount.mockResolvedValue(100);
        redisService.getErrorCounts.mockResolvedValue({
            network: 1,
            auth: 2,
            'rate-limit': 3,
            rejection: 4,
        });

        const text = await service.getMetricsAsText();

        expect(text).toContain('mail_outbox_lag_seconds 45');
        expect(text).toContain('bullmq_queue_depth 10');
        // Success rate: 100 / (100 + 1 + 2 + 3 + 4) = 100 / 110 = 0.9091
        expect(text).toContain('smtp_delivery_success_rate 0.9091');
        expect(text).toContain('smtp_delivery_errors_total{category="network"} 1');
        expect(text).toContain('smtp_delivery_errors_total{category="auth"} 2');
        expect(text).toContain('smtp_delivery_errors_total{category="rate-limit"} 3');
        expect(text).toContain('smtp_delivery_errors_total{category="rejection"} 4');
    });

    it('should handle zero attempts and default success rate to 1.0000', async () => {
        outboxRepository.getOldestLagSeconds.mockResolvedValue(0);
        mockQueue.getJobCounts.mockResolvedValue({ waiting: 0, delayed: 0, paused: 0 });
        redisService.getSuccessCount.mockResolvedValue(0);
        redisService.getErrorCounts.mockResolvedValue({
            network: 0,
            auth: 0,
            'rate-limit': 0,
            rejection: 0,
        });

        const text = await service.getMetricsAsText();

        expect(text).toContain('mail_outbox_lag_seconds 0');
        expect(text).toContain('bullmq_queue_depth 0');
        expect(text).toContain('smtp_delivery_success_rate 1.0000');
    });

    it('should handle BullMQ job count check failure gracefully', async () => {
        outboxRepository.getOldestLagSeconds.mockResolvedValue(0);
        mockQueue.getJobCounts.mockRejectedValue(new Error('Redis connection lost'));
        redisService.getSuccessCount.mockResolvedValue(10);
        redisService.getErrorCounts.mockResolvedValue({
            network: 0,
            auth: 0,
            'rate-limit': 0,
            rejection: 0,
        });

        const text = await service.getMetricsAsText();

        expect(text).toContain('bullmq_queue_depth 0');
        expect(text).toContain('smtp_delivery_success_rate 1.0000');
    });
});
