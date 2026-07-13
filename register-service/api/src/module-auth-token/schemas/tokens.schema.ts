import { index, pgTable, text, timestamp, uniqueIndex, uuid, varchar } from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { sessions } from '@/module-auth-token/schemas/sessions.schema';

export const tokens = pgTable(
    'tokens',
    {
        id: uuid('token_id').primaryKey().defaultRandom(),
        sessionId: uuid('session_id')
            .notNull()
            .references(() => sessions.id, { onDelete: 'cascade' }),
        jti: varchar('jti').notNull(),
        refreshTokenHash: text('refresh_token_hash').notNull(),
        encryptedReplacementToken: text('encrypted_replacement_token'),
        expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
        revokedAt: timestamp('revoked_at', { withTimezone: true }),
        replacedByTokenId: uuid('replaced_by_token_id').references((): any => tokens.id, {
            onDelete: 'set null',
        }),
        replacedAt: timestamp('replaced_at', { withTimezone: true }),
        graceExpiresAt: timestamp('grace_expires_at', { withTimezone: true }),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
        updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        tokensJtiUniqueIdx: uniqueIndex('tokens_jti_unique_idx').on(table.jti),
        tokensSessionValidIdx: index('tokens_session_valid_idx')
            .on(table.sessionId, table.expiresAt)
            .where(sql`revoked_at is null`),
        tokensGraceIdx: index('tokens_grace_idx')
            .on(table.graceExpiresAt)
            .where(sql`replaced_at is not null and revoked_at is null`),
    }),
);

export type TokenSelect = typeof tokens.$inferSelect;
export type TokenInsert = typeof tokens.$inferInsert;
