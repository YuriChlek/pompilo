import { Body, Controller, Get, Post, Query, UseGuards } from '@nestjs/common';
import {
    AckIdentityOutboxEventsDto,
    ListIdentityOutboxEventsDto,
} from '@/module-integration/dto/identity-outbox.dto';
import { ServiceTokenGuard } from '@/module-integration/guards/service-token.guard';
import { IdentityOutboxRepository } from '@/module-integration/repository/identity-outbox.repository';

@Controller('integration/identity-events')
@UseGuards(ServiceTokenGuard)
export class IdentityOutboxController {
    constructor(private readonly identityOutboxRepository: IdentityOutboxRepository) {}

    @Get('pending')
    async listPending(@Query() query: ListIdentityOutboxEventsDto) {
        const events = await this.identityOutboxRepository.listPending(query.limit ?? 50);
        return {
            events: events.map(event => event.payload),
        };
    }

    @Post('ack')
    async ack(@Body() dto: AckIdentityOutboxEventsDto) {
        const acknowledged = await this.identityOutboxRepository.markPublished(dto.eventIds);
        return {
            acknowledged,
        };
    }
}
