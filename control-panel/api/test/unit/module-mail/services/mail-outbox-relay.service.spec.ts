import { Test, TestingModule } from '@nestjs/testing';
import { MailOutboxRelayService } from '@/module-mail/services/mail-outbox-relay.service';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { getQueueToken } from '@nestjs/bullmq';
import { MAIL_QUEUE } from '@/module-mail/constants/mail.constants';
import { MailOutboxSelect } from '@/module-mail/schemas/mail-outbox.schema';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { ConfigService } from '@nestjs/config';

type PgNotificationHandler = (message: { channel: string; payload?: string }) => void;
type PgErrorHandler = (error: Error) => void;
type MockPgClient = {
    connect: jest.Mock;
    query: jest.Mock;
    end: jest.Mock;
    on: jest.Mock;
    handlers: {
        notification?: PgNotificationHandler;
        error?: PgErrorHandler;
    };
};

const mockPgClients: MockPgClient[] = [];

jest.mock('pg', () => ({
    Client: jest.fn().mockImplementation(() => {
        const client: MockPgClient = {
            connect: jest.fn().mockResolvedValue(undefined),
            query: jest.fn().mockResolvedValue(undefined),
            end: jest.fn().mockResolvedValue(undefined),
            handlers: {},
            on: jest.fn(
                (
                    event: 'notification' | 'error',
                    handler: PgNotificationHandler | PgErrorHandler,
                ) => {
                    if (event === 'notification') {
                        client.handlers.notification = handler as PgNotificationHandler;
                    }
                    if (event === 'error') {
                        client.handlers.error = handler as PgErrorHandler;
                    }
                    return client;
                },
            ),
        };
        mockPgClients.push(client);
        return client;
    }),
}));

describe('MailOutboxRelayService', () => {
    let service: MailOutboxRelayService;
    let repository: {
        claimRecords: jest.Mock;
        markAsQueued: jest.Mock;
        markPublishFailedForRetry: jest.Mock;
        releaseStaleLocks: jest.Mock;
    };
    let mockQueue: {
        add: jest.Mock;
    };
    let readinessService: {
        setReadinessStatus: jest.Mock;
    };
    let mockConfigService: {
        get: jest.Mock;
        getOrThrow: jest.Mock;
    };

    beforeEach(async () => {
        jest.clearAllMocks();
        mockPgClients.length = 0;

        repository = {
            claimRecords: jest.fn(),
            markAsQueued: jest.fn().mockResolvedValue(true),
            markPublishFailedForRetry: jest.fn(),
            releaseStaleLocks: jest.fn(),
        };

        mockQueue = {
            add: jest.fn(),
        };

        readinessService = {
            setReadinessStatus: jest.fn(),
        };

        mockConfigService = {
            get: jest.fn().mockReturnValue({
                retryBackoffType: 'exponential',
                retryBackoffDelay: 2000,
                retryBackoffJitter: 0.2,
                retryAttempts: 3,
            }),
            getOrThrow: jest.fn((key: string) => {
                const values: Record<string, string> = {
                    DB_HOST: 'localhost',
                    DB_PORT: '5432',
                    DB_USER: 'admin',
                    DB_PASSWORD: 'admin_pass',
                    DB_NAME: 'pampilo_db',
                };

                return values[key];
            }),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailOutboxRelayService,
                { provide: MailOutboxRepository, useValue: repository },
                { provide: getQueueToken(MAIL_QUEUE), useValue: mockQueue },
                { provide: MailReadinessService, useValue: readinessService },
                { provide: ConfigService, useValue: mockConfigService },
            ],
        }).compile();

        service = module.get<MailOutboxRelayService>(MailOutboxRelayService);
    });

    afterEach(async () => {
        await service.onApplicationShutdown();
    });

    describe('PostgreSQL notifications', () => {
        it('should listen for mail_outbox_inserted and trigger relay immediately', async () => {
            const records: Partial<MailOutboxSelect>[] = [
                {
                    id: 'event-record-1',
                    idempotencyKey: 'event-key-1',
                    status: 'pending',
                    payloadJson: { to: 'a@b.com' },
                    queuedJobId: null,
                },
            ];
            repository.releaseStaleLocks.mockResolvedValue(0);
            repository.claimRecords.mockResolvedValue(records);
            mockQueue.add.mockResolvedValue({ id: 'event-job-1' });

            await service.onApplicationBootstrap();

            const client = mockPgClients[0];
            expect(client.connect).toHaveBeenCalled();
            expect(client.query).toHaveBeenCalledWith('LISTEN mail_outbox_inserted');

            client.handlers.notification?.({
                channel: 'mail_outbox_inserted',
                payload: 'event-record-1',
            });
            await new Promise(resolve => setImmediate(resolve));

            expect(repository.claimRecords).toHaveBeenCalledWith(50, expect.any(String));
            expect(mockQueue.add).toHaveBeenCalledWith(
                'send-mail',
                expect.objectContaining({ to: 'a@b.com', idempotencyKey: 'event-key-1' }),
                expect.objectContaining({ jobId: 'event-key-1' }),
            );
            expect(repository.markAsQueued).toHaveBeenCalledWith(
                'event-record-1',
                'event-job-1',
                expect.any(String),
            );
        });
    });

    describe('handleStaleLocks', () => {
        it('should call releaseStaleLocks and log warning if reclaimed > 0', async () => {
            repository.releaseStaleLocks.mockResolvedValue(2);
            await service.handleStaleLocks();
            expect(repository.releaseStaleLocks).toHaveBeenCalledWith(5, 3);
        });

        it('should skip if already running', async () => {
            repository.releaseStaleLocks.mockImplementation(
                () => new Promise(resolve => setTimeout(() => resolve(0), 100)),
            );

            const p1 = service.handleStaleLocks();
            const p2 = service.handleStaleLocks();

            await Promise.all([p1, p2]);
            expect(repository.releaseStaleLocks).toHaveBeenCalledTimes(1);
        });

        it('should handle repository errors safely', async () => {
            repository.releaseStaleLocks.mockRejectedValue(new Error('DB error'));
            await expect(service.handleStaleLocks()).resolves.not.toThrow();
            expect(repository.releaseStaleLocks).toHaveBeenCalled();
        });
    });

    describe('handleCron', () => {
        it('should claim records and add them to BullMQ', async () => {
            const records: Partial<MailOutboxSelect>[] = [
                {
                    id: '1',
                    idempotencyKey: 'key1',
                    status: 'pending',
                    payloadJson: { to: 'a@b.com' },
                    queuedJobId: null,
                },
            ];
            repository.claimRecords.mockResolvedValue(records);
            mockQueue.add.mockResolvedValue({ id: 'job-1', remove: jest.fn() });
            repository.markAsQueued.mockResolvedValue(true);

            await service.handleCron();

            expect(repository.claimRecords).toHaveBeenCalledWith(50, expect.any(String));
            expect(mockQueue.add).toHaveBeenCalledWith(
                'send-mail',
                expect.objectContaining({ to: 'a@b.com', idempotencyKey: 'key1' }),
                {
                    jobId: 'key1',
                    attempts: 3,
                    backoff: {
                        type: 'exponential',
                        delay: 2000,
                        jitter: 0.2,
                    },
                    removeOnComplete: {
                        age: 3600,
                        count: 1000,
                    },
                    removeOnFail: expect.any(Object) as object,
                },
            );

            expect(repository.markAsQueued).toHaveBeenCalledWith('1', 'job-1', expect.any(String));
        });

        it('should skip processing if already running', async () => {
            repository.claimRecords.mockImplementation(
                () => new Promise(resolve => setTimeout(() => resolve([]), 100)),
            );

            // Call twice simultaneously
            const p1 = service.handleCron();
            const p2 = service.handleCron();

            await Promise.all([p1, p2]);

            expect(repository.claimRecords).toHaveBeenCalledTimes(1);
        });

        it('should skip publishing if record already has queuedJobId', async () => {
            const records: Partial<MailOutboxSelect>[] = [
                {
                    id: '2',
                    idempotencyKey: 'key2',
                    status: 'queued',
                    queuedJobId: 'existing-job',
                    payloadJson: { to: 'a@b.com' },
                },
            ];
            repository.claimRecords.mockResolvedValue(records);

            await service.handleCron();

            expect(mockQueue.add).not.toHaveBeenCalled();
            expect(repository.markAsQueued).not.toHaveBeenCalled();
        });

        it('should handle BullMQ errors safely and signal unhealthy status', async () => {
            const records: Partial<MailOutboxSelect>[] = [
                {
                    id: '3',
                    idempotencyKey: 'key3',
                    status: 'pending',
                    payloadJson: { to: 'a@b.com' },
                    queuedJobId: null,
                },
            ];
            repository.claimRecords.mockResolvedValue(records);
            mockQueue.add.mockRejectedValue(new Error('Redis is down'));

            await service.handleCron();

            // Should try to add
            expect(mockQueue.add).toHaveBeenCalled();
            // Should not mark as queued
            expect(repository.markAsQueued).not.toHaveBeenCalled();
            expect(repository.markPublishFailedForRetry).toHaveBeenCalledWith(
                '3',
                'Redis is down',
                3,
            );
            // Should signal unhealthy status
            expect(readinessService.setReadinessStatus).toHaveBeenCalledWith(
                'unhealthy',
                expect.stringContaining('Relay publish failed'),
                600,
            );
        });

        it('should pass payloadEncrypted directly to BullMQ', async () => {
            const records: Partial<MailOutboxSelect>[] = [
                {
                    id: '4',
                    idempotencyKey: 'key4',
                    status: 'pending',
                    payloadEncrypted: 'enc:data',
                    payloadJson: null,
                    queuedJobId: null,
                },
            ];
            repository.claimRecords.mockResolvedValue(records);
            mockQueue.add.mockResolvedValue({ id: 'job-4', remove: jest.fn() });
            repository.markAsQueued.mockResolvedValue(true);

            await service.handleCron();

            expect(mockQueue.add).toHaveBeenCalledWith(
                'send-mail',
                expect.objectContaining({ payloadEncrypted: 'enc:data', idempotencyKey: 'key4' }),
                {
                    jobId: 'key4',
                    attempts: 3,
                    backoff: {
                        type: 'exponential',
                        delay: 2000,
                        jitter: 0.2,
                    },
                    removeOnComplete: {
                        age: 3600,
                        count: 1000,
                    },
                    removeOnFail: expect.any(Object) as object,
                },
            );

            expect(repository.markAsQueued).toHaveBeenCalledWith('4', 'job-4', expect.any(String));
        });

        it('should execute draining loop up to safety limit when full batch is processed', async () => {
            const fullBatch = Array.from({ length: 50 }, (_, i) => ({
                id: `${i}`,
                idempotencyKey: `key-${i}`,
                status: 'pending' as const,
                payloadJson: { to: 'a@b.com' },
                queuedJobId: null,
                priority: 0,
                attemptCount: 0,
                lastError: null,
                availableAt: new Date(),
                lockedAt: null,
                lockedBy: null,
                queuedAt: null,
                createdAt: new Date(),
                updatedAt: new Date(),
            }));

            repository.claimRecords.mockResolvedValue(fullBatch);
            mockQueue.add.mockResolvedValue({ id: 'job-id', remove: jest.fn() });
            repository.markAsQueued.mockResolvedValue(true);

            await service.handleCron();

            expect(repository.claimRecords).toHaveBeenCalledTimes(10);
        });

        it('should remove the BullMQ job and release the record when queue confirmation fails', async () => {
            const remove = jest.fn().mockResolvedValue(undefined);
            repository.claimRecords.mockResolvedValue([
                {
                    id: '5',
                    idempotencyKey: 'key5',
                    status: 'pending',
                    payloadJson: { to: 'a@b.com' },
                    queuedJobId: null,
                },
            ]);
            repository.markAsQueued.mockRejectedValue(new Error('DB unavailable'));
            mockQueue.add.mockResolvedValue({ id: 'job-5', remove });

            await service.handleCron();

            expect(remove).toHaveBeenCalled();
            expect(repository.markPublishFailedForRetry).toHaveBeenCalledWith(
                '5',
                'DB unavailable',
                3,
            );
        });
    });
});
