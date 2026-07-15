import { Test, TestingModule } from '@nestjs/testing';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { PgDialect } from 'drizzle-orm/pg-core';
import * as schema from '@/module-drizzle/schemas';
import { createRepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';

describe('MailOutboxRepository', () => {
    let repository: MailOutboxRepository;
    let mockDb: {
        insert: jest.Mock;
        values: jest.Mock;
        returning: jest.Mock;
        execute: jest.Mock;
    };

    beforeEach(async () => {
        mockDb = {
            insert: jest.fn().mockReturnThis(),
            values: jest.fn().mockReturnThis(),
            returning: jest.fn().mockResolvedValue([{ id: 'test-id' }]),
            execute: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [MailOutboxRepository, { provide: DRIZZLE_PROVIDER, useValue: mockDb }],
        }).compile();

        repository = module.get<MailOutboxRepository>(MailOutboxRepository);
    });

    it('should create an outbox record with payloadJson', async () => {
        const payload = {
            idempotencyKey: 'key1',
            payloadJson: { to: 'test@test.com', templateName: 'status_update' },
            priority: 0,
        };

        await repository.create(payload);

        expect(mockDb.values).toHaveBeenCalledWith(
            expect.objectContaining({
                idempotencyKey: 'key1',
                payloadJson: payload.payloadJson,
                status: 'pending',
                attemptCount: 0,
            }),
        );
    });

    it('should create an outbox record with payloadEncrypted', async () => {
        const payload = {
            idempotencyKey: 'key2',
            payloadEncrypted: 'encrypted:data:here',
            priority: 0,
        };

        await repository.create(payload);

        expect(mockDb.insert).toHaveBeenCalled();
        expect(mockDb.values).toHaveBeenCalledWith(
            expect.objectContaining({
                idempotencyKey: 'key2',
                payloadEncrypted: payload.payloadEncrypted,
            }),
        );
    });

    it('should throw invariant error if both payloads are provided', async () => {
        const payload = {
            idempotencyKey: 'key3',
            payloadJson: { to: 'test' },
            payloadEncrypted: 'encrypted',
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Both payloadJson and payloadEncrypted are provided.',
        );
    });

    it('should throw invariant error if neither payload is provided', async () => {
        const payload = {
            idempotencyKey: 'key4',
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Neither payloadJson nor payloadEncrypted is provided.',
        );
    });

    it('should throw invariant error if payloadJson contains obvious secret (token)', async () => {
        const payload = {
            idempotencyKey: 'key5',
            payloadJson: { text: 'Your reset token=abcdef1234567890abcdef' },
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Secret-bearing payload detected in payloadJson. Use payloadEncrypted instead.',
        );
    });

    it('should throw invariant error if payloadJson contains obvious secret (verification code)', async () => {
        const payload = {
            idempotencyKey: 'key6',
            payloadJson: { text: 'Your verification code is: 123456' },
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Secret-bearing payload detected in payloadJson. Use payloadEncrypted instead.',
        );
    });

    it('should throw invariant error if payloadJson contains forbidden key (html)', async () => {
        const payload = {
            idempotencyKey: 'key-forbidden-html',
            payloadJson: { to: 'test', html: '<h1>Hello</h1>' },
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Secret-bearing payload detected in payloadJson. Use payloadEncrypted instead.',
        );
    });

    it('should throw invariant error if payloadJson contains forbidden key (code)', async () => {
        const payload = {
            idempotencyKey: 'key-forbidden-code',
            payloadJson: { to: 'test', code: '1234' },
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Secret-bearing payload detected in payloadJson. Use payloadEncrypted instead.',
        );
    });

    it('should throw invariant error if payloadJson contains rendered body', async () => {
        const payload = {
            idempotencyKey: 'key-forbidden-body',
            payloadJson: { to: 'test', body: 'Rendered email body must be encrypted' },
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Secret-bearing payload detected in payloadJson. Use payloadEncrypted instead.',
        );
    });

    it('should throw invariant error if payloadJson contains nested token field', async () => {
        const payload = {
            idempotencyKey: 'key-nested-token',
            payloadJson: {
                to: 'test',
                metadata: {
                    resetToken: 'abcdef1234567890abcdef',
                },
            },
            priority: 0,
        };

        await expect(repository.create(payload)).rejects.toThrow(
            'MailOutbox invariant violation: Secret-bearing payload detected in payloadJson. Use payloadEncrypted instead.',
        );
    });

    it('should allow transaction client to be passed', async () => {
        const mockInsert = jest.fn().mockReturnThis();
        const txClient = {
            insert: mockInsert,
            values: jest.fn().mockReturnThis(),
            returning: jest.fn().mockResolvedValue([{ id: 'tx-id' }]),
        } as unknown as NodePgDatabase<typeof schema>;

        const payload = {
            idempotencyKey: 'key7',
            payloadJson: { to: 'test' },
            priority: 0,
        };

        await repository.create(payload, createRepositoryTransaction(txClient));

        expect(mockInsert).toHaveBeenCalled();
        expect(mockDb.insert).not.toHaveBeenCalled();
    });

    describe('releaseStaleLocks', () => {
        it('should execute the stale locks query and return row count', async () => {
            mockDb.execute.mockResolvedValue({ rowCount: 5 });

            const reclaimed = await repository.releaseStaleLocks(5, 3);

            expect(mockDb.execute).toHaveBeenCalled();
            const executeCalls = mockDb.execute.mock.calls as unknown[][];
            // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
            const query = new PgDialect().sqlToQuery(executeCalls[0][0] as any);
            expect(query.sql).toContain('AND queued_job_id IS NULL');
            expect(query.sql).toContain('make_interval(mins => $1::integer)');
            expect(query.sql).toContain(`'failed'::"public"."mail_outbox_status_enum"`);
            expect(query.sql).toContain(`'pending'::"public"."mail_outbox_status_enum"`);
            expect(query.sql).toContain('mail_outbox.attempt_count + 1 >= $2::integer');
            expect(query.params).toEqual([5, 3, 3]);
            expect(reclaimed).toBe(5);
        });

        it('should return 0 if rowCount is null', async () => {
            mockDb.execute.mockResolvedValue({});

            const reclaimed = await repository.releaseStaleLocks(5, 3);

            expect(mockDb.execute).toHaveBeenCalled();
            expect(reclaimed).toBe(0);
        });
    });

    describe('markPublishFailedForRetry', () => {
        it('should execute an immediate retry release query', async () => {
            mockDb.execute.mockResolvedValue({ rowCount: 1 });

            await repository.markPublishFailedForRetry('outbox-1', 'Redis is down', 3);

            expect(mockDb.execute).toHaveBeenCalled();
            const executeCalls = mockDb.execute.mock.calls as unknown[][];
            // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
            const query = new PgDialect().sqlToQuery(executeCalls[0][0] as any);
            expect(query.sql).toContain(`ELSE 'pending'::"public"."mail_outbox_status_enum"`);
            expect(query.sql).toContain('attempt_count + 1 >= $1::integer');
            expect(query.sql).toContain('make_interval(mins => POWER(2, attempt_count)::integer)');
            expect(query.sql).toContain('AND queued_job_id IS NULL');
            expect(query.params).toEqual([3, 'Redis is down', 3, 'outbox-1']);
        });
    });

    describe('queue delivery state', () => {
        it('should mark only the record owned by the relay instance as queued', async () => {
            mockDb.execute.mockResolvedValue({ rowCount: 1 });

            await expect(repository.markAsQueued('outbox-1', 'job-1', 'relay-1')).resolves.toBe(
                true,
            );

            const executeCalls = mockDb.execute.mock.calls as unknown[][];
            // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
            const query = new PgDialect().sqlToQuery(executeCalls[0][0] as any);
            expect(query.sql).toContain('AND locked_by = $3');
            expect(query.sql).toContain('AND queued_job_id IS NULL');
            expect(query.params).toEqual(['job-1', 'outbox-1', 'relay-1']);
        });

        it('should report whether a job is confirmed for delivery', async () => {
            mockDb.execute.mockResolvedValue({ rows: [{ '?column?': 1 }] });

            await expect(repository.isQueuedForDelivery('job-1')).resolves.toBe(true);

            const executeCalls = mockDb.execute.mock.calls as unknown[][];
            // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
            const query = new PgDialect().sqlToQuery(executeCalls[0][0] as any);
            expect(query.sql).toContain(`status = 'queued'::"public"."mail_outbox_status_enum"`);
            expect(query.params).toEqual(['job-1']);
        });
    });

    describe('deleteOldRecordsByStatusInBatches', () => {
        it('should execute delete query and handle chunked deletes in loops', async () => {
            mockDb.execute
                .mockResolvedValueOnce({ rowCount: 1000 })
                .mockResolvedValueOnce({ rowCount: 500 });

            const olderThan = new Date();
            const totalDeleted = await repository.deleteOldRecordsByStatusInBatches(
                'sent',
                olderThan,
                1000,
                0, // delayMs = 0 to avoid test waiting
            );

            expect(totalDeleted).toBe(1500);
            expect(mockDb.execute).toHaveBeenCalledTimes(2);

            const executeCalls = mockDb.execute.mock.calls as unknown[][];
            // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
            const query = new PgDialect().sqlToQuery(executeCalls[0][0] as any);
            expect(query.sql).toContain('DELETE FROM mail_outbox');
            expect(query.sql).toContain('WHERE status = $1::mail_outbox_status_enum');
            expect(query.params).toEqual(['sent', olderThan.toISOString(), 1000]);
        });
    });
});
