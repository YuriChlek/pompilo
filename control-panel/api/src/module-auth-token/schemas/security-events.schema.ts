import {
    check,
    index,
    integer,
    jsonb,
    pgTable,
    text,
    timestamp,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { users } from '@/module-user/schemas';
import { knownDevices } from '@/module-auth-token/schemas/known-devices.schema';
import { sessions } from '@/module-auth-token/schemas/sessions.schema';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';

export const securityEvents = pgTable(
    'security_events',
    {
        id: uuid('security_event_id').primaryKey().defaultRandom(),
        userId: uuid('user_id').references(() => users.id, { onDelete: 'set null' }),
        realm: varchar('realm', { length: 255 }).notNull(),
        sessionId: uuid('session_id').references(() => sessions.id, { onDelete: 'set null' }),
        knownDeviceId: uuid('known_device_id').references(() => knownDevices.id, {
            onDelete: 'set null',
        }),
        eventType: varchar('event_type', { length: 255 }).$type<SecurityEventType>().notNull(),
        riskScore: integer('risk_score').notNull().default(0),
        riskReason: text('risk_reason'),
        ipAddress: varchar('ip_address', { length: 45 }),
        country: varchar('country', { length: 2 }),
        region: varchar('region', { length: 128 }),
        city: varchar('city', { length: 128 }),
        userAgent: text('user_agent'),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
        metadata: jsonb('metadata'),
    },
    table => ({
        securityEventsRealmCheck: check(
            'security_events_realm_check',
            sql`${table.realm} in ('customer', 'admin')`,
        ),
        securityEventsDeviceIdx: index('security_events_device_idx')
            .on(table.knownDeviceId)
            .where(sql`known_device_id is not null`),
        securityEventsUserOccurredIdx: index('security_events_user_occurred_idx').on(
            table.userId,
            table.realm,
            table.createdAt,
        ),
    }),
);

export type SecurityEventSelect = typeof securityEvents.$inferSelect;
export type SecurityEventInsert = typeof securityEvents.$inferInsert;
