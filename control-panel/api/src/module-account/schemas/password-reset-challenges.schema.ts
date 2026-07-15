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

export const passwordResetChallenges = pgTable(
    'password_reset_challenges',
    {
        id: uuid('password_reset_challenge_id').primaryKey().defaultRandom(),
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        selector: varchar('selector', { length: 64 }).notNull().unique(),
        verifierDigest: text('verifier_digest').notNull(),
        attemptCount: integer('attempt_count').notNull().default(0),
        lockedUntil: timestamp('locked_until', { withTimezone: true }),
        expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
        usedAt: timestamp('used_at', { withTimezone: true }),
        invalidatedAt: timestamp('invalidated_at', { withTimezone: true }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        activeChallengePerUserIdx: uniqueIndex('active_password_reset_challenge_per_user_idx')
            .on(table.userId)
            .where(sql`used_at IS NULL AND invalidated_at IS NULL`),
        passwordResetUserIdIdx: index('password_reset_user_id_idx').on(table.userId),
        passwordResetExpiresAtIdx: index('password_reset_expires_at_idx').on(table.expiresAt),
        passwordResetCompositeIdx: index('password_reset_composite_idx').on(
            table.userId,
            table.usedAt,
            table.expiresAt,
        ),
    }),
);

export type PasswordResetChallengeSelect = typeof passwordResetChallenges.$inferSelect;
export type PasswordResetChallengeInsert = typeof passwordResetChallenges.$inferInsert;
