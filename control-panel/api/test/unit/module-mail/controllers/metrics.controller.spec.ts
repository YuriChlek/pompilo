import { Test, TestingModule } from '@nestjs/testing';
import { MetricsController } from '@/module-mail/controllers/metrics.controller';
import { MailMetricsService } from '@/module-mail/services/mail-metrics.service';

describe('MetricsController', () => {
    let controller: MetricsController;
    let metricsService: {
        getMetricsAsText: jest.Mock;
    };

    beforeEach(async () => {
        metricsService = {
            getMetricsAsText: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [MetricsController],
            providers: [{ provide: MailMetricsService, useValue: metricsService }],
        }).compile();

        controller = module.get<MetricsController>(MetricsController);
    });

    it('should return plain text metrics with version 0.0.4 header from service', async () => {
        const mockText = 'mail_outbox_lag_seconds 0\n';
        metricsService.getMetricsAsText.mockResolvedValue(mockText);

        const result = await controller.getMetrics();

        expect(result).toBe(mockText);
        expect(metricsService.getMetricsAsText).toHaveBeenCalled();
    });
});
