import { Logger } from '@nestjs/common';
import { ConsoleMailService } from '@/module-mail/services/console-mail.service';

describe('ConsoleMailService', () => {
    it('should not log recipients, subject, or rendered email body', async () => {
        const loggerSpy = jest.spyOn(Logger.prototype, 'log').mockImplementation();
        const service = new ConsoleMailService();

        await service.createDeliveryRequest({
            to: 'secret-recipient@example.com',
            subject: 'Secret subject',
            html: '<p>Secret token abc123</p>',
            text: 'Secret token abc123',
        });

        const loggedValue = loggerSpy.mock.calls.flat().join(' ');
        expect(loggedValue).toContain('Recipient count: 1');
        expect(loggedValue).not.toContain('secret-recipient@example.com');
        expect(loggedValue).not.toContain('Secret subject');
        expect(loggedValue).not.toContain('Secret token abc123');

        loggerSpy.mockRestore();
    });
});
