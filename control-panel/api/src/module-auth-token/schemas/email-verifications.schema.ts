import { index, pgTable, text, timestamp, uniqueIndex, uuid } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { users } from '@/module-user/schemas';

export const emailVerifications = pgTable(
    'email_verifications',
    {
        id: uuid('email_verification_id').primaryKey().defaultRandom(),
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        tokenHash: text('token_hash').notNull(),
        expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
        consumedAt: timestamp('consumed_at', { withTimezone: true }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        emailVerificationsTokenHashUnique: uniqueIndex('email_verifications_token_hash_unique').on(
            table.tokenHash,
        ),
        emailVerificationsExpiryIdx: index('email_verifications_expiry_idx')
            .on(table.expiresAt)
            .where(sql`consumed_at is null`),
    }),
);

export type EmailVerificationSelect = typeof emailVerifications.$inferSelect;
export type EmailVerificationInsert = typeof emailVerifications.$inferInsert;
