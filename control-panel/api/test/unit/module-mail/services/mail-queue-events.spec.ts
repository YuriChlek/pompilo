import { Test, TestingModule } from '@nestjs/testing';
import { MailQueueEvents } from '@/module-mail/events/mail-queue-events';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { getQueueToken } from '@nestjs/bullmq';
import { MAIL_POISON_FAILURE_CODES, MAIL_QUEUE } from '@/module-mail/constants/mail.constants';

describe('MailQueueEvents', () => {
    let service: MailQueueEvents;
    let readinessService: { setReadinessStatus: jest.Mock };
    let repository: { markAsFailed: jest.Mock };
    let mockQueue: { getJob: jest.Mock };

    beforeEach(async () => {
        readinessService = { setReadinessStatus: jest.fn() };
        repository = { markAsFailed: jest.fn() };
        mockQueue = { getJob: jest.fn() };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailQueueEvents,
                { provide: MailReadinessService, useValue: readinessService },
                { provide: MailOutboxRepository, useValue: repository },
                { provide: getQueueToken(MAIL_QUEUE), useValue: mockQueue },
            ],
        }).compile();

        service = module.get<MailQueueEvents>(MailQueueEvents);
    });

    it('should update readiness and log completion on completed event', async () => {
        await service.onCompleted({ jobId: 'job-1', returnvalue: 'ok' });
        expect(readinessService.setReadinessStatus).toHaveBeenCalledWith(
            'healthy',
            undefined,
            3600,
        );
    });

    it('should retry job without degrading global readiness if attempts are not exhausted', async () => {
        mockQueue.getJob.mockResolvedValue({
            opts: { attempts: 3 },
            attemptsMade: 1,
        });

        await service.onFailed({ jobId: 'job-1', failedReason: 'SMTP network error' });

        expect(readinessService.setReadinessStatus).not.toHaveBeenCalled();
        expect(repository.markAsFailed).not.toHaveBeenCalled();
    });

    it('should immediately fail outbox record if failure is unrecoverable (decryption failure)', async () => {
        mockQueue.getJob.mockResolvedValue({
            opts: { attempts: 3 },
            attemptsMade: 1,
        });

        await service.onFailed({ jobId: 'job-1', failedReason: 'decryption_failure: wrong key' });

        expect(readinessService.setReadinessStatus).not.toHaveBeenCalled();
        expect(repository.markAsFailed).toHaveBeenCalledWith(
            'job-1',
            'decryption_failure: wrong key',
        );
    });

    it('should immediately fail outbox record if failure is unrecoverable (invalid payload schema)', async () => {
        mockQueue.getJob.mockResolvedValue({
            opts: { attempts: 3 },
            attemptsMade: 1,
        });

        await service.onFailed({
            jobId: 'job-1',
            failedReason: MAIL_POISON_FAILURE_CODES.INVALID_PAYLOAD_SCHEMA,
        });

        expect(readinessService.setReadinessStatus).not.toHaveBeenCalled();
        expect(repository.markAsFailed).toHaveBeenCalledWith(
            'job-1',
            MAIL_POISON_FAILURE_CODES.INVALID_PAYLOAD_SCHEMA,
        );
    });

    it('should isolate forbidden plaintext payload without degrading global readiness', async () => {
        mockQueue.getJob.mockResolvedValue({
            opts: { attempts: 3 },
            attemptsMade: 1,
        });

        await service.onFailed({
            jobId: 'job-1',
            failedReason: MAIL_POISON_FAILURE_CODES.PLAINTEXT_PAYLOAD_FORBIDDEN,
        });

        expect(readinessService.setReadinessStatus).not.toHaveBeenCalled();
        expect(repository.markAsFailed).toHaveBeenCalledWith(
            'job-1',
            MAIL_POISON_FAILURE_CODES.PLAINTEXT_PAYLOAD_FORBIDDEN,
        );
    });

    it('should fail outbox record if attempts are exhausted', async () => {
        mockQueue.getJob.mockResolvedValue({
            opts: { attempts: 3 },
            attemptsMade: 3,
        });

        await service.onFailed({ jobId: 'job-1', failedReason: 'SMTP connection failed' });

        expect(repository.markAsFailed).toHaveBeenCalledWith('job-1', 'SMTP connection failed');
    });
});
