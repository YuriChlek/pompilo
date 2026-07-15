import { Controller, Get, Header } from '@nestjs/common';
import { MailMetricsService } from '@/module-mail/services/mail-metrics.service';
import { SkipResponseEnvelope } from '@/common/decorators/skip-response-envelope.decorator';

@Controller('metrics')
export class MetricsController {
    constructor(private readonly mailMetricsService: MailMetricsService) {}

    @Get()
    @Header('Content-Type', 'text/plain; version=0.0.4; charset=utf-8')
    @SkipResponseEnvelope()
    async getMetrics(): Promise<string> {
        return await this.mailMetricsService.getMetricsAsText();
    }
}
