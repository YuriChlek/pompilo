import { Logger } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import { SmtpMailService } from '@/module-mail/services/smtp-mail.service';
import { MailSettingsRepository } from '@/module-mail/repository/mail-settings.repository';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import { MailRedisInvalidationService } from '@/module-mail/services/mail-redis-invalidation.service';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import * as nodemailer from 'nodemailer';
import { MailSettingsSelect } from '@/module-mail/schemas';

import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';

jest.mock('nodemailer');

describe('SmtpMailService', () => {
    let service: SmtpMailService;
    let repository: {
        findSingleton: jest.Mock;
    };
    let settingsService: {
        getCachedSettings: jest.Mock;
    };
    let outboxRepository: {
        create: jest.Mock;
    };
    let encryptionService: {
        decryptMailSecret: jest.Mock;
        encryptMailSecret: jest.Mock;
    };
    let redisService: {
        incrementFailureCount: jest.Mock;
        logDeliveryError: jest.Mock;
        logSuccessfulDelivery: jest.Mock;
        incrementSuccessCount: jest.Mock;
        incrementErrorCount: jest.Mock;
    };
    let invalidationService: {
        onInvalidate: jest.Mock;
    };
    let readinessService: {
        setReadinessStatus: jest.Mock;
    };
    let mockTransport: {
        verify: jest.Mock;
        sendMail: jest.Mock;
        close: jest.Mock;
    };

    beforeEach(async () => {
        repository = {
            findSingleton: jest.fn(),
        };
        settingsService = {
            getCachedSettings: jest
                .fn()
                .mockImplementation(
                    (tx?: unknown) => repository.findSingleton(tx) as Promise<unknown>,
                ),
        };
        outboxRepository = {
            create: jest.fn().mockResolvedValue({
                id: 'outbox-id',
                idempotencyKey: 'idemp-key',
                status: 'pending',
            }),
        };
        encryptionService = {
            decryptMailSecret: jest.fn(),
            encryptMailSecret: jest.fn().mockReturnValue('encrypted-payload'),
        };
        redisService = {
            incrementFailureCount: jest.fn(),
            logDeliveryError: jest.fn(),
            logSuccessfulDelivery: jest.fn(),
            incrementSuccessCount: jest.fn(),
            incrementErrorCount: jest.fn(),
        };
        invalidationService = {
            onInvalidate: jest.fn(),
        };
        readinessService = {
            setReadinessStatus: jest.fn(),
        };
        mockTransport = {
            verify: jest.fn(),
            sendMail: jest.fn(),
            close: jest.fn(),
        };

        (nodemailer.createTransport as jest.Mock).mockClear();
        (nodemailer.createTransport as jest.Mock).mockReturnValue(mockTransport);

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                SmtpMailService,
                { provide: MailSettingsRepository, useValue: repository },
                { provide: MailOutboxRepository, useValue: outboxRepository },
                { provide: MailEncryptionService, useValue: encryptionService },
                { provide: MailRedisService, useValue: redisService },
                { provide: MailRedisInvalidationService, useValue: invalidationService },
                { provide: MailReadinessService, useValue: readinessService },
                { provide: MailSettingsService, useValue: settingsService },
            ],
        }).compile();

        service = module.get<SmtpMailService>(SmtpMailService);
    });

    it('should create and verify transport successfully', async () => {
        const settings = {
            enabled: true,
            smtpHost: 'localhost',
            smtpPort: 1025,
            smtpSecure: false,
            smtpUser: 'user',
            smtpPasswordEncrypted: 'encrypted',
            fromName: 'Test',
            fromAddress: 'test@test.com',
        } as MailSettingsSelect;
        repository.findSingleton.mockResolvedValue(settings);
        encryptionService.decryptMailSecret.mockReturnValue('decrypted');
        mockTransport.verify.mockResolvedValue(true);

        await service.sendMailDirect({
            to: 'to@test.com',
            subject: 'Subject',
            text: 'text',
            html: 'html',
        });

        expect(nodemailer.createTransport).toHaveBeenCalledWith({
            host: 'localhost',
            port: 1025,
            secure: false,
            auth: { user: 'user', pass: 'decrypted' },
        });
        expect(mockTransport.verify).toHaveBeenCalled();
        expect(mockTransport.sendMail).toHaveBeenCalledWith(
            expect.objectContaining({
                to: 'to@test.com',
                subject: 'Subject',
            }),
        );
    });

    it('should never write a full recipient address to delivery logs', async () => {
        const logSpy = jest.spyOn(Logger.prototype, 'log').mockImplementation(() => undefined);
        repository.findSingleton.mockResolvedValue({
            enabled: true,
            smtpHost: 'localhost',
            smtpPort: 1025,
            smtpSecure: false,
            fromName: 'Test',
            fromAddress: 'sender@example.com',
        } as MailSettingsSelect);
        mockTransport.verify.mockResolvedValue(true);
        mockTransport.sendMail.mockResolvedValue({ accepted: ['sensitive.user@example.com'] });

        await service.sendMailDirect({
            to: 'sensitive.user@example.com',
            subject: 'Subject',
            text: 'text',
            html: 'html',
        });

        const logged = logSpy.mock.calls.flat().join(' ');
        expect(logged).toContain('s***@e***.com');
        expect(logged).not.toContain('sensitive.user@example.com');
        logSpy.mockRestore();
    });

    it('should cache transport and reuse it if config is same', async () => {
        const settings = {
            enabled: true,
            smtpHost: 'localhost',
            smtpPort: 1025,
            smtpSecure: false,
        } as MailSettingsSelect;
        repository.findSingleton.mockResolvedValue(settings);
        mockTransport.verify.mockResolvedValue(true);

        await service.sendMailDirect({ to: '1@test.com', subject: 'S', text: 't', html: 'h' });
        await service.sendMailDirect({ to: '2@test.com', subject: 'S', text: 't', html: 'h' });

        expect(nodemailer.createTransport).toHaveBeenCalledTimes(1);
    });

    it('should reset transport when invalidation callback is called', async () => {
        const settings = {
            enabled: true,
            smtpHost: 'localhost',
            smtpPort: 1025,
        } as MailSettingsSelect;
        repository.findSingleton.mockResolvedValue(settings);
        mockTransport.verify.mockResolvedValue(true);

        // First call creates transport
        await service.sendMailDirect({ to: '1@test.com', subject: 'S', text: 't', html: 'h' });

        // Trigger invalidation (manually call the callback passed to onInvalidate)
        const invalidateCalls = invalidationService.onInvalidate.mock.calls as unknown[][];
        const invalidateCallback = invalidateCalls[0][0] as () => void;
        invalidateCallback();

        // Second call should recreate transport
        await service.sendMailDirect({ to: '2@test.com', subject: 'S', text: 't', html: 'h' });

        expect(nodemailer.createTransport).toHaveBeenCalledTimes(2);
    });

    it('should handle SMTP errors and increment failure count', async () => {
        repository.findSingleton.mockResolvedValue({
            enabled: true,
            smtpHost: 'h',
            smtpPort: 1,
            fromName: 'n',
            fromAddress: 'a',
        } as MailSettingsSelect);
        mockTransport.verify.mockResolvedValue(true);
        const connError = new Error('Connection timed out');
        Object.defineProperty(connError, 'code', { value: 'ETIMEDOUT' });
        mockTransport.sendMail.mockRejectedValue(connError);
        redisService.incrementFailureCount.mockResolvedValue(1);

        await expect(
            service.sendMailDirect({ to: 'to', subject: 's', text: 't', html: 'h' }),
        ).rejects.toThrow('Connection timed out');

        expect(redisService.incrementFailureCount).toHaveBeenCalled();
        expect(redisService.logDeliveryError).toHaveBeenCalledWith(
            'smtp_send_failed',
            'Connection timed out',
        );
        expect(readinessService.setReadinessStatus).not.toHaveBeenCalled();
    });

    it('should verify transport with plaintext password', async () => {
        const settings = {
            smtpHost: 'h',
            smtpPort: 1,
            smtpSecure: false,
            smtpUser: 'u',
            smtpPassword: 'p',
        };
        mockTransport.verify.mockResolvedValue(true);

        await service.verifyTransport(settings);

        expect(nodemailer.createTransport).toHaveBeenCalledWith(
            expect.objectContaining({
                auth: { user: 'u', pass: 'p' },
            }),
        );
        expect(mockTransport.verify).toHaveBeenCalled();
    });

    it('should verify transport with encrypted password', async () => {
        const settings = {
            smtpHost: 'h',
            smtpPort: 1,
            smtpSecure: false,
            smtpUser: 'u',
            smtpPasswordEncrypted: 'enc',
        };
        encryptionService.decryptMailSecret.mockReturnValue('dec');
        mockTransport.verify.mockResolvedValue(true);

        await service.verifyTransport(settings);

        expect(encryptionService.decryptMailSecret).toHaveBeenCalledWith('enc');
        expect(nodemailer.createTransport).toHaveBeenCalledWith(
            expect.objectContaining({
                auth: { user: 'u', pass: 'dec' },
            }),
        );
    });

    it('should trigger circuit breaker if failure count reaches 5', async () => {
        repository.findSingleton.mockResolvedValue({
            enabled: true,
            smtpHost: 'h',
            smtpPort: 1,
            fromName: 'n',
            fromAddress: 'a',
        } as MailSettingsSelect);
        mockTransport.verify.mockResolvedValue(true);
        const connError = new Error('ECONNREFUSED');
        Object.defineProperty(connError, 'code', { value: 'ECONNREFUSED' });
        mockTransport.sendMail.mockRejectedValue(connError);
        redisService.incrementFailureCount.mockResolvedValue(5);

        await expect(
            service.sendMailDirect({ to: 'to', subject: 's', text: 't', html: 'h' }),
        ).rejects.toThrow('ECONNREFUSED');

        expect(readinessService.setReadinessStatus).toHaveBeenCalledWith(
            'unhealthy',
            'too_many_delivery_failures',
            120,
        );
    });

    it('should not increment failure count or trigger circuit breaker on non-connection/non-auth errors', async () => {
        repository.findSingleton.mockResolvedValue({
            enabled: true,
            smtpHost: 'h',
            smtpPort: 1,
            fromName: 'n',
            fromAddress: 'a',
        } as MailSettingsSelect);
        mockTransport.verify.mockResolvedValue(true);
        mockTransport.sendMail.mockRejectedValue(new Error('Invalid recipient format'));

        await expect(
            service.sendMailDirect({ to: 'to', subject: 's', text: 't', html: 'h' }),
        ).rejects.toThrow('Invalid recipient format');

        expect(redisService.incrementFailureCount).not.toHaveBeenCalled();
        expect(redisService.logDeliveryError).toHaveBeenCalledWith(
            'smtp_send_error',
            'Invalid recipient format',
        );
        expect(readinessService.setReadinessStatus).not.toHaveBeenCalled();
    });

    it('should trigger circuit breaker if transport verification fails', async () => {
        repository.findSingleton.mockResolvedValue({
            enabled: true,
            smtpHost: 'h',
            smtpPort: 1,
            fromName: 'n',
            fromAddress: 'a',
        } as MailSettingsSelect);
        mockTransport.verify.mockRejectedValue(new Error('Auth failed'));

        await expect(
            service.sendMailDirect({ to: 'to', subject: 's', text: 't', html: 'h' }),
        ).rejects.toThrow('smtp_verification_failed');

        expect(readinessService.setReadinessStatus).toHaveBeenCalledWith(
            'unhealthy',
            'smtp_verification_failed',
            600,
        );
    });
});
