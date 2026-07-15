import { Inject, Injectable } from '@nestjs/common';
import { NodePgDatabase } from 'drizzle-orm/node-postgres';
import { sql, eq } from 'drizzle-orm';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import {
    getTransactionClient,
    RepositoryTransaction,
} from '@/module-drizzle/repository/transaction.repository';
import * as schema from '@/module-drizzle/schemas';
import {
    mailOutbox,
    MailOutboxInsert,
    MailOutboxSelect,
} from '@/module-mail/schemas/mail-outbox.schema';

@Injectable()
export class MailOutboxRepository {
    constructor(
        @Inject(DRIZZLE_PROVIDER)
        private readonly db: NodePgDatabase<typeof schema>,
    ) {}

    async claimRecords(limit: number, lockedBy: string): Promise<MailOutboxSelect[]> {
        const now = new Date();

        const result = await this.db.execute(sql`
            UPDATE mail_outbox
            SET locked_at = ${now.toISOString()},
                locked_by = ${lockedBy},
                updated_at = ${now.toISOString()}
            WHERE mail_outbox_id IN (
                SELECT mail_outbox_id
                FROM mail_outbox
                WHERE status = 'pending'
                  AND locked_at IS NULL
                  AND available_at <= ${now.toISOString()}
                  AND queued_job_id IS NULL
                ORDER BY priority DESC, available_at ASC
                LIMIT ${limit}
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *;
        `);

        return result.rows.map(row => ({
            id: row.mail_outbox_id as string,
            idempotencyKey: row.idempotency_key as string,
            status: row.status as typeof mailOutbox.$inferSelect.status,
            payloadEncrypted: row.payload_encrypted ? (row.payload_encrypted as string) : null,
            payloadJson: row.payload_json,
            priority: Number(row.priority),
            attemptCount: Number(row.attempt_count),
            lastError: row.last_error ? (row.last_error as string) : null,
            availableAt: row.available_at ? new Date(row.available_at as string) : new Date(),
            lockedAt: row.locked_at ? new Date(row.locked_at as string) : null,
            lockedBy: row.locked_by ? (row.locked_by as string) : null,
            queuedJobId: row.queued_job_id ? (row.queued_job_id as string) : null,
            queuedAt: row.queued_at ? new Date(row.queued_at as string) : null,
            createdAt: row.created_at ? new Date(row.created_at as string) : new Date(),
            updatedAt: row.updated_at ? new Date(row.updated_at as string) : new Date(),
        })) as MailOutboxSelect[];
    }

    async markAsQueued(id: string, jobId: string, lockedBy: string): Promise<boolean> {
        const result = await this.db.execute(sql`
            UPDATE mail_outbox
            SET status = 'queued'::"public"."mail_outbox_status_enum",
                queued_job_id = ${jobId},
                queued_at = NOW(),
                locked_at = NULL,
                locked_by = NULL,
                updated_at = NOW()
            WHERE mail_outbox_id = ${id}
              AND status = 'pending'::"public"."mail_outbox_status_enum"
              AND locked_by = ${lockedBy}
              AND queued_job_id IS NULL
            RETURNING mail_outbox_id;
        `);

        return (result.rowCount ?? 0) === 1;
    }

    async isQueuedForDelivery(jobId: string): Promise<boolean> {
        const result = await this.db.execute(sql`
            SELECT 1
            FROM mail_outbox
            WHERE status = 'queued'::"public"."mail_outbox_status_enum"
              AND queued_job_id = ${jobId}
            LIMIT 1;
        `);

        return result.rows.length === 1;
    }

    async releaseStaleLocks(staleThresholdMinutes: number, maxAttempts: number): Promise<number> {
        const result = await this.db.execute(sql`
            WITH stale_records AS (
                SELECT mail_outbox_id, attempt_count
                FROM mail_outbox
                WHERE status = 'pending'
                  AND locked_at IS NOT NULL
                  AND locked_at < NOW() - make_interval(mins => ${staleThresholdMinutes}::integer)
                  AND queued_job_id IS NULL
            )
            UPDATE mail_outbox
            SET
                status = CASE
                    WHEN mail_outbox.attempt_count + 1 >= ${maxAttempts}::integer THEN 'failed'::"public"."mail_outbox_status_enum"
                    ELSE 'pending'::"public"."mail_outbox_status_enum"
                END,
                attempt_count = mail_outbox.attempt_count + 1,
                last_error = 'Stale lock reclaimed due to timeout',
                locked_at = NULL,
                locked_by = NULL,
                updated_at = NOW(),
                available_at = CASE
                    WHEN mail_outbox.attempt_count + 1 >= ${maxAttempts}::integer THEN mail_outbox.available_at
                    ELSE NOW() + make_interval(mins => POWER(2, mail_outbox.attempt_count)::integer)
                END
            FROM stale_records
            WHERE mail_outbox.mail_outbox_id = stale_records.mail_outbox_id
            RETURNING mail_outbox.mail_outbox_id;
        `);

        return result.rowCount ?? 0;
    }

    async markPublishFailedForRetry(id: string, error: string, maxAttempts: number): Promise<void> {
        await this.db.execute(sql`
            UPDATE mail_outbox
            SET
                status = CASE
                    WHEN attempt_count + 1 >= ${maxAttempts}::integer THEN 'failed'::"public"."mail_outbox_status_enum"
                    ELSE 'pending'::"public"."mail_outbox_status_enum"
                END,
                attempt_count = attempt_count + 1,
                last_error = ${error},
                locked_at = NULL,
                locked_by = NULL,
                updated_at = NOW(),
                available_at = CASE
                    WHEN attempt_count + 1 >= ${maxAttempts}::integer THEN available_at
                    ELSE NOW() + make_interval(mins => POWER(2, attempt_count)::integer)
                END
            WHERE mail_outbox_id = ${id}
              AND queued_job_id IS NULL;
        `);
    }

    async markAsSent(jobId: string): Promise<void> {
        await this.db
            .update(mailOutbox)
            .set({
                status: 'sent',
                updatedAt: new Date(),
            })
            .where(eq(mailOutbox.queuedJobId, jobId));
    }

    async markAsFailed(jobId: string, error: string): Promise<void> {
        await this.db
            .update(mailOutbox)
            .set({
                status: 'failed',
                lastError: error,
                updatedAt: new Date(),
            })
            .where(eq(mailOutbox.queuedJobId, jobId));
    }

    async deleteOldRecordsByStatusInBatches(
        status: 'sent' | 'failed',
        olderThan: Date,
        batchSize = 1000,
        delayMs = 150,
    ): Promise<number> {
        let totalDeleted = 0;
        while (true) {
            const result = await this.db.execute(sql`
                WITH target_records AS (
                    SELECT mail_outbox_id
                    FROM mail_outbox
                    WHERE status = ${status}::mail_outbox_status_enum
                      AND created_at < ${olderThan.toISOString()}
                    LIMIT ${batchSize}
                )
                DELETE FROM mail_outbox
                WHERE mail_outbox_id IN (SELECT mail_outbox_id FROM target_records)
                RETURNING mail_outbox_id;
            `);

            const deletedCount = result.rowCount ?? 0;
            totalDeleted += deletedCount;

            if (deletedCount < batchSize) {
                break;
            }

            await new Promise(resolve => setTimeout(resolve, delayMs));
        }
        return totalDeleted;
    }

    async create(
        data: Omit<
            MailOutboxInsert,
            | 'id'
            | 'status'
            | 'attemptCount'
            | 'createdAt'
            | 'updatedAt'
            | 'availableAt'
            | 'lockedAt'
            | 'lockedBy'
            | 'queuedJobId'
            | 'queuedAt'
            | 'lastError'
        >,
        transaction?: RepositoryTransaction,
    ): Promise<MailOutboxSelect> {
        this.validatePayloadInvariant(data);

        const dbClient = getTransactionClient(transaction, this.db);
        const [outbox] = await dbClient
            .insert(mailOutbox)
            .values({
                ...data,
                status: 'pending',
                attemptCount: 0,
            })
            .returning();

        return outbox;
    }

    private validatePayloadInvariant(data: Partial<MailOutboxInsert>): void {
        const hasJson = data.payloadJson !== undefined && data.payloadJson !== null;
        const hasEncrypted = data.payloadEncrypted !== undefined && data.payloadEncrypted !== null;

        if (hasJson && hasEncrypted) {
            throw new Error(
                'MailOutbox invariant violation: Both payloadJson and payloadEncrypted are provided.',
            );
        }

        if (!hasJson && !hasEncrypted) {
            throw new Error(
                'MailOutbox invariant violation: Neither payloadJson nor payloadEncrypted is provided.',
            );
        }

        if (hasJson) {
            const payloadJson = this.parsePayloadJsonValue(data.payloadJson);
            if (this.containsPlaintextSecret(payloadJson)) {
                throw new Error(
                    'MailOutbox invariant violation: Secret-bearing payload detected in payloadJson. Use payloadEncrypted instead.',
                );
            }
        }
    }

    private parsePayloadJsonValue(value: unknown): unknown {
        if (typeof value !== 'string') {
            return value;
        }

        try {
            return JSON.parse(value);
        } catch {
            return value;
        }
    }

    private containsPlaintextSecret(value: unknown): boolean {
        if (typeof value === 'string') {
            return this.containsSecretPattern(value);
        }

        if (Array.isArray(value)) {
            return value.some(item => this.containsPlaintextSecret(item));
        }

        if (!value || typeof value !== 'object') {
            return false;
        }

        return Object.entries(value as Record<string, unknown>).some(([key, nestedValue]) => {
            if (this.isForbiddenPlaintextPayloadKey(key)) {
                return true;
            }

            return this.containsPlaintextSecret(nestedValue);
        });
    }

    private isForbiddenPlaintextPayloadKey(key: string): boolean {
        const normalizedKey = key.toLowerCase().replace(/[_-]/g, '');

        return (
            ['html', 'text', 'body', 'code', 'token', 'otp', 'password', 'secret'].includes(
                normalizedKey,
            ) ||
            normalizedKey.endsWith('token') ||
            normalizedKey.includes('password') ||
            normalizedKey.includes('secret') ||
            normalizedKey.includes('credential') ||
            normalizedKey.includes('verificationcode') ||
            normalizedKey.includes('resetlink')
        );
    }

    private containsSecretPattern(value: string): boolean {
        return (
            /(?:token|code|otp)=\s*[a-zA-Z0-9_-]{6,}/i.test(value) ||
            /verification code is:?\s*<?[a-zA-Z0-9_-]{4,}>?/i.test(value) ||
            /reset(?:\s|-)?password.*[?&]token=/i.test(value)
        );
    }

    async getOldestLagSeconds(): Promise<number> {
        const result = await this.db.execute(sql`
            SELECT EXTRACT(EPOCH FROM (NOW() - created_at))::integer as lag
            FROM mail_outbox
            WHERE status IN ('pending', 'failed')
            ORDER BY created_at ASC
            LIMIT 1;
        `);

        if (result.rows.length === 0) {
            return 0;
        }

        return Number(result.rows[0].lag ?? 0);
    }
}
