import { Test, TestingModule } from '@nestjs/testing';
import { Job } from 'bullmq';
import { ConfigService } from '@nestjs/config';
import { MailProcessorService } from '@/module-mail/services/mail-processor.service';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { SmtpMailService } from '@/module-mail/services/smtp-mail.service';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { MailHealth } from '@/module-mail/interfaces/mail-service.interface';
import { MAIL_POISON_FAILURE_CODES } from '@/module-mail/constants/mail.constants';

describe('MailProcessorService', () => {
    let service: MailProcessorService;
    let mailReadinessService: { getHealth: jest.Mock; setReadinessStatus: jest.Mock };
    let smtpMailService: { sendMailDirect: jest.Mock };
    let mailEncryptionService: { decryptMailSecret: jest.Mock };
    let mailOutboxRepository: {
        isQueuedForDelivery: jest.Mock;
        markAsSent: jest.Mock;
        markAsFailed: jest.Mock;
    };
    let configService: { get: jest.Mock };

    beforeEach(async () => {
        mailReadinessService = { getHealth: jest.fn(), setReadinessStatus: jest.fn() };
        smtpMailService = { sendMailDirect: jest.fn() };
        mailEncryptionService = { decryptMailSecret: jest.fn() };
        mailOutboxRepository = {
            isQueuedForDelivery: jest.fn().mockResolvedValue(true),
            markAsSent: jest.fn(),
            markAsFailed: jest.fn(),
        };
        configService = {
            get: jest.fn().mockReturnValue({
                workerConcurrency: 4,
                queueLimiterMax: 100,
                queueLimiterDuration: 1000,
            }),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailProcessorService,
                { provide: MailReadinessService, useValue: mailReadinessService },
                { provide: SmtpMailService, useValue: smtpMailService },
                { provide: MailEncryptionService, useValue: mailEncryptionService },
                { provide: MailOutboxRepository, useValue: mailOutboxRepository },
                { provide: ConfigService, useValue: configService },
            ],
        }).compile();

        service = module.get<MailProcessorService>(MailProcessorService);
    });

    it('should throw error if health is not healthy', async () => {
        mailReadinessService.getHealth.mockResolvedValue({
            status: 'unhealthy',
            lastError: 'some error',
        } as MailHealth);

        const job = { id: '1', data: {} } as unknown as Job;

        await expect(service.process(job)).rejects.toThrow(
            'Mail system is not healthy: some error',
        );
        expect(smtpMailService.sendMailDirect).not.toHaveBeenCalled();
    });

    it('should not send a job before the outbox record is confirmed as queued', async () => {
        mailOutboxRepository.isQueuedForDelivery.mockResolvedValue(false);

        const job = { id: '1', data: {} } as unknown as Job;

        await expect(service.process(job)).rejects.toThrow(
            'Outbox record is not confirmed for delivery',
        );
        expect(mailReadinessService.getHealth).not.toHaveBeenCalled();
        expect(smtpMailService.sendMailDirect).not.toHaveBeenCalled();
    });

    it('should reject plaintext delivery payload without retry', async () => {
        mailReadinessService.getHealth.mockResolvedValue({ status: 'healthy' } as MailHealth);

        const payload = {
            to: 'test@example.com',
            subject: 'Test',
            html: '<b>hi</b>',
            text: 'hi',
        };
        const job = { id: '1', data: payload } as unknown as Job;

        await expect(service.process(job)).rejects.toThrow(
            MAIL_POISON_FAILURE_CODES.PLAINTEXT_PAYLOAD_FORBIDDEN,
        );

        expect(smtpMailService.sendMailDirect).not.toHaveBeenCalled();
        expect(mailOutboxRepository.markAsFailed).toHaveBeenCalledWith(
            '1',
            MAIL_POISON_FAILURE_CODES.PLAINTEXT_PAYLOAD_FORBIDDEN,
        );
        expect(mailEncryptionService.decryptMailSecret).not.toHaveBeenCalled();
        expect(mailOutboxRepository.markAsSent).not.toHaveBeenCalled();
    });

    it('should decrypt and process encrypted payload', async () => {
        mailReadinessService.getHealth.mockResolvedValue({ status: 'healthy' } as MailHealth);

        const payload = {
            to: 'secure@example.com',
            subject: 'Secure',
            html: '<b>secure</b>',
            text: 'secure',
        };
        mailEncryptionService.decryptMailSecret.mockReturnValue(JSON.stringify(payload));

        const job = { id: '1', data: { payloadEncrypted: 'enc:data' } } as unknown as Job;

        await service.process(job);

        expect(mailEncryptionService.decryptMailSecret).toHaveBeenCalledWith('enc:data');
        expect(smtpMailService.sendMailDirect).toHaveBeenCalledWith(
            expect.objectContaining({ to: 'secure@example.com', subject: 'Secure' }),
        );
        expect(mailReadinessService.setReadinessStatus).toHaveBeenCalledWith('healthy');
        expect(mailOutboxRepository.markAsSent).toHaveBeenCalledWith('1');
    });

    it('should reject undecryptable payload without retry and store safe failure code', async () => {
        mailReadinessService.getHealth.mockResolvedValue({ status: 'healthy' } as MailHealth);

        mailEncryptionService.decryptMailSecret.mockImplementation(() => {
            throw new Error('decryption error');
        });

        const job = { id: '1', data: { payloadEncrypted: 'enc:data' } } as unknown as Job;

        await expect(service.process(job)).rejects.toThrow(
            MAIL_POISON_FAILURE_CODES.DECRYPTION_FAILURE,
        );

        expect(smtpMailService.sendMailDirect).not.toHaveBeenCalled();
        expect(mailOutboxRepository.markAsFailed).toHaveBeenCalledWith(
            '1',
            MAIL_POISON_FAILURE_CODES.DECRYPTION_FAILURE,
        );
        expect(mailOutboxRepository.markAsSent).not.toHaveBeenCalled();
    });

    it('should reject invalid payload schema without retry and store safe failure code', async () => {
        mailReadinessService.getHealth.mockResolvedValue({ status: 'healthy' } as MailHealth);

        mailEncryptionService.decryptMailSecret.mockReturnValue(
            JSON.stringify({ to: 'test@test.com' }),
        );
        const job = { id: '1', data: { payloadEncrypted: 'enc:data' } } as unknown as Job;

        await expect(service.process(job)).rejects.toThrow(
            MAIL_POISON_FAILURE_CODES.INVALID_PAYLOAD_SCHEMA,
        );

        expect(smtpMailService.sendMailDirect).not.toHaveBeenCalled();
        expect(mailOutboxRepository.markAsFailed).toHaveBeenCalledWith(
            '1',
            MAIL_POISON_FAILURE_CODES.INVALID_PAYLOAD_SCHEMA,
        );
        expect(mailOutboxRepository.markAsSent).not.toHaveBeenCalled();
    });

    it('should not block normal queue processing after rejecting a poison job', async () => {
        mailReadinessService.getHealth.mockResolvedValue({ status: 'healthy' } as MailHealth);

        const poisonJob = {
            id: 'poison-1',
            data: { payloadEncrypted: 'poison' },
        } as unknown as Job;
        const validJob = {
            id: 'valid-1',
            data: { payloadEncrypted: 'valid' },
        } as unknown as Job;
        mailEncryptionService.decryptMailSecret
            .mockReturnValueOnce(JSON.stringify({ subject: 'Missing recipient' }))
            .mockReturnValueOnce(
                JSON.stringify({
                    to: 'test@example.com',
                    subject: 'Test',
                    html: '<b>hi</b>',
                    text: 'hi',
                }),
            );

        await expect(service.process(poisonJob)).rejects.toThrow(
            MAIL_POISON_FAILURE_CODES.INVALID_PAYLOAD_SCHEMA,
        );
        await service.process(validJob);

        expect(mailOutboxRepository.markAsFailed).toHaveBeenCalledWith(
            'poison-1',
            MAIL_POISON_FAILURE_CODES.INVALID_PAYLOAD_SCHEMA,
        );
        expect(smtpMailService.sendMailDirect).toHaveBeenCalledWith(
            expect.objectContaining({ to: 'test@example.com', subject: 'Test' }),
        );
        expect(mailOutboxRepository.markAsSent).toHaveBeenCalledWith('valid-1');
    });

    it('should propagate errors from sendMailDirect', async () => {
        mailReadinessService.getHealth.mockResolvedValue({ status: 'healthy' } as MailHealth);

        const payload = {
            to: 'test@example.com',
            subject: 'Test',
            html: 'hi',
            text: 'hi',
        };
        mailEncryptionService.decryptMailSecret.mockReturnValue(JSON.stringify(payload));
        const job = { id: '1', data: { payloadEncrypted: 'enc:data' } } as unknown as Job;

        smtpMailService.sendMailDirect.mockRejectedValue(new Error('SMTP connection failed'));

        await expect(service.process(job)).rejects.toThrow('SMTP connection failed');
        expect(mailReadinessService.setReadinessStatus).toHaveBeenCalledWith(
            'unhealthy',
            'smtp_connection_failed',
            120,
        );
    });

    it('should set worker concurrency on application bootstrap', () => {
        const mockWorker = {
            concurrency: 1,
            opts: {},
            run: jest.fn().mockResolvedValue(undefined),
        };
        Object.defineProperty(service, 'worker', {
            value: mockWorker as unknown,
            writable: true,
        });

        service.onApplicationBootstrap();

        expect(mockWorker.concurrency).toBe(4);
        expect(mockWorker.opts).toEqual({
            limiter: {
                max: 100,
                duration: 1000,
            },
        });
        expect(mockWorker.run).toHaveBeenCalled();
    });

    describe('Circuit Breaker Worker Pausing', () => {
        let mockWorker: { pause: jest.Mock; resume: jest.Mock; isPaused: jest.Mock };

        beforeEach(() => {
            jest.useFakeTimers();
            mockWorker = {
                pause: jest.fn().mockResolvedValue(undefined),
                resume: jest.fn().mockResolvedValue(undefined),
                isPaused: jest.fn().mockReturnValue(false),
            };
            Object.defineProperty(service, 'worker', {
                value: mockWorker as any,
                writable: true,
                configurable: true,
            });
        });

        afterEach(() => {
            jest.useRealTimers();
        });

        it('should pause worker and schedule resume when health check fails with too_many_delivery_failures', async () => {
            mailReadinessService.getHealth.mockResolvedValue({
                status: 'unhealthy',
                lastError: 'too_many_delivery_failures',
            } as MailHealth);

            const job = { id: '1', data: {} } as unknown as Job;

            await expect(service.process(job)).rejects.toThrow(
                'Mail system is not healthy: too_many_delivery_failures',
            );

            expect(mockWorker.pause).toHaveBeenCalled();

            // Advance timers by 60 seconds
            jest.advanceTimersByTime(60000);

            expect(mockWorker.resume).toHaveBeenCalled();
        });

        it('should pause worker and schedule resume when SMTP send failure trips the circuit breaker', async () => {
            // First call during readiness check returns healthy
            // Second call after failure (postSendHealth) returns unhealthy with too_many_delivery_failures
            mailReadinessService.getHealth
                .mockResolvedValueOnce({ status: 'healthy' } as MailHealth)
                .mockResolvedValueOnce({
                    status: 'unhealthy',
                    lastError: 'too_many_delivery_failures',
                } as MailHealth);

            smtpMailService.sendMailDirect.mockRejectedValue(new Error('SMTP Error'));

            const payload = {
                to: 'test@example.com',
                subject: 'Test',
                html: 'hi',
                text: 'hi',
            };
            mailEncryptionService.decryptMailSecret.mockReturnValue(JSON.stringify(payload));
            const job = { id: '1', data: { payloadEncrypted: 'enc:data' } } as unknown as Job;

            await expect(service.process(job)).rejects.toThrow('SMTP Error');

            expect(mockWorker.pause).toHaveBeenCalled();

            // Advance timers by 60 seconds
            jest.advanceTimersByTime(60000);

            expect(mockWorker.resume).toHaveBeenCalled();
        });
    });
});
