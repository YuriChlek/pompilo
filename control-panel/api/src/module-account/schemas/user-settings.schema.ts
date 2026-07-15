import { boolean, pgTable, timestamp, uuid, varchar } from 'drizzle-orm/pg-core';
import { users } from '@/module-user/schemas/users.schema';

export const userSettings = pgTable('user_settings', {
    userId: uuid('user_id')
        .primaryKey()
        .references(() => users.id, { onDelete: 'cascade' }),
    timezone: varchar('timezone', { length: 100 }).notNull().default('UTC'),
    notifySecurityAlerts: boolean('notify_security_alerts').notNull().default(true),
    createdAt: timestamp('created_at').notNull().defaultNow(),
    updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

export type UserSettingsSelect = typeof userSettings.$inferSelect;
export type UserSettingsInsert = typeof userSettings.$inferInsert;
