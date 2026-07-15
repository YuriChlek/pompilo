import {
    check,
    index,
    pgTable,
    text,
    timestamp,
    uniqueIndex,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { users } from '@/module-user/schemas';
import { sessions } from '@/module-auth-token/schemas/sessions.schema';

export const reauthConfirmations = pgTable(
    'reauth_confirmations',
    {
        id: uuid('reauth_confirmation_id').primaryKey().defaultRandom(),
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        realm: varchar('realm', { length: 255 }).notNull(),
        sessionId: uuid('session_id')
            .notNull()
            .references(() => sessions.id, { onDelete: 'cascade' }),
        actionScope: varchar('action_scope', { length: 255 }).notNull(),
        confirmationTokenHash: text('confirmation_token_hash').notNull(),
        expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
        consumedAt: timestamp('consumed_at', { withTimezone: true }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        reauthConfirmationsRealmCheck: check(
            'reauth_confirmations_realm_check',
            sql`${table.realm} in ('customer', 'admin')`,
        ),
        reauthConfirmationsTokenHashUnique: uniqueIndex(
            'reauth_confirmations_token_hash_unique',
        ).on(table.confirmationTokenHash),
        reauthConfirmationsExpiryIdx: index('reauth_confirmations_expiry_idx')
            .on(table.expiresAt)
            .where(sql`consumed_at is null`),
    }),
);

export type ReauthConfirmationSelect = typeof reauthConfirmations.$inferSelect;
export type ReauthConfirmationInsert = typeof reauthConfirmations.$inferInsert;

export const isReauthConfirmationActive = (
    confirmation: ReauthConfirmationSelect,
    now = new Date(),
): boolean => {
    return confirmation.consumedAt === null && confirmation.expiresAt.getTime() > now.getTime();
};
