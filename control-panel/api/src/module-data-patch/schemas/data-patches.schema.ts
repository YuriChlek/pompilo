import { integer, pgTable, text, timestamp } from 'drizzle-orm/pg-core';

export const dataPatches = pgTable('data_patches', {
    patchName: text('patch_name').primaryKey(),
    checksum: text('checksum').notNull(),
    appliedAt: timestamp('applied_at', { withTimezone: true }).notNull().defaultNow(),
    durationMs: integer('duration_ms').notNull(),
    description: text('description'),
    appVersion: text('app_version'),
    nodeEnv: text('node_env'),
});

export type DataPatchSelect = typeof dataPatches.$inferSelect;
export type DataPatchInsert = typeof dataPatches.$inferInsert;
