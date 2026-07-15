import { index, jsonb, pgTable, timestamp, uuid, varchar } from 'drizzle-orm/pg-core';
import { users } from '@/module-user/schemas/users.schema';
import type {
    MailAuditAction,
    MailAuditPayload,
} from '@/module-mail/interfaces/mail-audit-event.interfaces';

export const mailAuditEvents = pgTable(
    'mail_audit_events',
    {
        id: uuid('mail_audit_event_id').primaryKey().defaultRandom(),
        action: varchar('action', { length: 100 }).$type<MailAuditAction>().notNull(),
        adminUserId: uuid('admin_user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        payload: jsonb('payload').$type<MailAuditPayload>().notNull(),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        mailAuditEventsActionCreatedAtIdx: index('mail_audit_events_action_created_at_idx').on(
            table.action,
            table.createdAt.desc(),
        ),
        mailAuditEventsAdminUserIdCreatedAtIdx: index(
            'mail_audit_events_admin_user_id_created_at_idx',
        ).on(table.adminUserId, table.createdAt.desc()),
    }),
);

export type MailAuditEventSelect = typeof mailAuditEvents.$inferSelect;
export type MailAuditEventInsert = typeof mailAuditEvents.$inferInsert;
