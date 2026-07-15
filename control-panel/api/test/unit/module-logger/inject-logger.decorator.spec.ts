import { Injectable } from '@nestjs/common';
import { Test } from '@nestjs/testing';
import { InjectLogger } from '@/module-logger/decorators/inject-logger.decorator';
import { getLoggerChannelToken } from '@/module-logger/tokens/logger.tokens';
import type { RuntimeLogger } from '@/module-logger/interfaces/logger.interfaces';

@Injectable()
class MailLoggerConsumer {
    constructor(@InjectLogger('mail') readonly logger: RuntimeLogger) {}
}

describe('InjectLogger', () => {
    it('injects the logger registered for the requested channel token', async () => {
        const mailLogger = {
            log: jest.fn(),
        };
        const moduleRef = await Test.createTestingModule({
            providers: [
                MailLoggerConsumer,
                {
                    provide: getLoggerChannelToken('mail'),
                    useValue: mailLogger,
                },
            ],
        }).compile();

        const consumer = moduleRef.get(MailLoggerConsumer);

        expect(consumer.logger).toBe(mailLogger);
    });
});
