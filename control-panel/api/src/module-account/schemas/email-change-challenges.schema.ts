import {
    index,
    integer,
    pgTable,
    text,
    timestamp,
    uniqueIndex,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { users } from '@/module-user/schemas/users.schema';

export const emailChangeChallenges = pgTable(
    'email_change_challenges',
    {
        id: uuid('email_change_challenge_id').primaryKey().defaultRandom(),
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        newEmail: varchar('new_email', { length: 255 }).notNull(),
        codeDigest: text('code_digest').notNull(),
        attemptCount: integer('attempt_count').notNull().default(0),
        lockedUntil: timestamp('locked_until', { withTimezone: true }),
        expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
        usedAt: timestamp('used_at', { withTimezone: true }),
        invalidatedAt: timestamp('invalidated_at', { withTimezone: true }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        activeEmailChangeChallengePerUserIdx: uniqueIndex(
            'active_email_change_challenge_per_user_idx',
        )
            .on(table.userId)
            .where(sql`used_at IS NULL AND invalidated_at IS NULL`),
        emailChangeUserIdIdx: index('email_change_user_id_idx').on(table.userId),
        emailChangeExpiresAtIdx: index('email_change_expires_at_idx').on(table.expiresAt),
        emailChangeCompositeIdx: index('email_change_composite_idx').on(
            table.userId,
            table.usedAt,
            table.expiresAt,
        ),
    }),
);

export type EmailChangeChallengeSelect = typeof emailChangeChallenges.$inferSelect;
export type EmailChangeChallengeInsert = typeof emailChangeChallenges.$inferInsert;
