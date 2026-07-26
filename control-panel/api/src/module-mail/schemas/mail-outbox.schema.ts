import {
    check,
    index,
    integer,
    jsonb,
    pgEnum,
    pgTable,
    text,
    timestamp,
    uniqueIndex,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';

export const mailOutboxStatusEnum = pgEnum('mail_outbox_status_enum', [
    'pending',
    'queued',
    'sent',
    'failed',
]);

export const mailOutbox = pgTable(
    'mail_outbox',
    {
        id: uuid('mail_outbox_id').primaryKey().defaultRandom(),
        idempotencyKey: varchar('idempotency_key', { length: 255 }).notNull(),
        status: mailOutboxStatusEnum('status').notNull().default('pending'),
        payloadEncrypted: text('payload_encrypted'),
        payloadJson: jsonb('payload_json'),
        priority: integer('priority').notNull().default(0),
        attemptCount: integer('attempt_count').notNull().default(0),
        lastError: text('last_error'),
        availableAt: timestamp('available_at', { withTimezone: true }).notNull().defaultNow(),
        lockedAt: timestamp('locked_at', { withTimezone: true }),
        lockedBy: varchar('locked_by', { length: 255 }),
        queuedJobId: varchar('queued_job_id', { length: 255 }),
        queuedAt: timestamp('queued_at', { withTimezone: true }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
        updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        mailOutboxIdempotencyKeyIdx: uniqueIndex('mail_outbox_idempotency_key_idx').on(
            table.idempotencyKey,
        ),
        mailOutboxStatusIdx: index('mail_outbox_status_idx').on(table.status),
        mailOutboxClaimIdx: index('mail_outbox_claim_idx')
            .on(table.priority.desc(), table.availableAt.asc())
            .where(sql`status = 'pending'`),
        mailOutboxStaleReclaimIdx: index('mail_outbox_stale_reclaim_idx')
            .on(table.lockedAt)
            .where(sql`status = 'pending' AND locked_at IS NOT NULL`),
        mailOutboxPayloadExactlyOneCheck: check(
            'mail_outbox_payload_exactly_one_check',
            sql`(payload_encrypted IS NOT NULL) <> (payload_json IS NOT NULL)`,
        ),
    }),
);

export type MailOutboxSelect = typeof mailOutbox.$inferSelect;
export type MailOutboxInsert = typeof mailOutbox.$inferInsert;
