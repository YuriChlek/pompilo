import {
    index,
    integer,
    jsonb,
    pgEnum,
    pgTable,
    timestamp,
    uniqueIndex,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';

export const identityOutboxEventStatusEnum = pgEnum('identity_outbox_event_status_enum', [
    'pending',
    'published',
    'failed',
]);

export const identityOutboxEvents = pgTable(
    'identity_outbox_events',
    {
        id: uuid('identity_outbox_event_id').primaryKey().defaultRandom(),
        eventId: uuid('event_id').notNull(),
        eventType: varchar('event_type', { length: 128 }).notNull(),
        eventVersion: integer('event_version').notNull().default(1),
        aggregateId: uuid('aggregate_id').notNull(),
        tenantId: uuid('tenant_id').notNull(),
        idempotencyKey: varchar('idempotency_key', { length: 255 }).notNull(),
        status: identityOutboxEventStatusEnum('status').notNull().default('pending'),
        payload: jsonb('payload').notNull(),
        availableAt: timestamp('available_at', { withTimezone: true }).notNull().defaultNow(),
        publishedAt: timestamp('published_at', { withTimezone: true }),
        lastError: varchar('last_error', { length: 1024 }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
        updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        identityOutboxEventIdIdx: uniqueIndex('identity_outbox_event_id_idx').on(table.eventId),
        identityOutboxIdempotencyKeyIdx: uniqueIndex('identity_outbox_idempotency_key_idx').on(
            table.idempotencyKey,
        ),
        identityOutboxStatusIdx: index('identity_outbox_status_idx').on(table.status),
        identityOutboxAvailableIdx: index('identity_outbox_available_idx').on(
            table.status,
            table.availableAt,
        ),
        identityOutboxTenantIdx: index('identity_outbox_tenant_idx').on(table.tenantId),
    }),
);

export type IdentityOutboxEventSelect = typeof identityOutboxEvents.$inferSelect;
export type IdentityOutboxEventInsert = typeof identityOutboxEvents.$inferInsert;
