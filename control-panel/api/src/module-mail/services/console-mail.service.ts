import { Injectable, Logger } from '@nestjs/common';
import { randomUUID } from 'crypto';
import {
    MailDeliveryRequest,
    MailService,
    SendMailPayload,
} from '@/module-mail/interfaces/mail-service.interface';

@Injectable()
export class ConsoleMailService implements MailService {
    private readonly logger = new Logger(ConsoleMailService.name);

    async createDeliveryRequest(payload: SendMailPayload): Promise<MailDeliveryRequest> {
        await Promise.resolve();
        const recipientCount = Array.isArray(payload.to) ? payload.to.length : 1;
        this.logger.log(`[MAIL MOCK ACCEPTED] Recipient count: ${recipientCount}`);

        return {
            outboxId: randomUUID(),
            idempotencyKey: randomUUID(),
            status: 'accepted',
        };
    }

    async verifyTransport(): Promise<void> {
        this.logger.log('[MAIL MOCK VERIFY] Transport verified (always success)');
        return Promise.resolve();
    }
}
