import { boolean, integer, pgTable, text, timestamp, uuid, varchar } from 'drizzle-orm/pg-core';
import { users } from '@/module-user/schemas/users.schema';

export const mailSettings = pgTable('mail_settings', {
    id: uuid('mail_settings_id').primaryKey().defaultRandom(),
    singletonKey: boolean('singleton_key').notNull().default(true).unique(),
    provider: varchar('provider', { length: 32 }).notNull().default('smtp'),
    smtpHost: varchar('smtp_host', { length: 255 }).notNull(),
    smtpPort: integer('smtp_port').notNull(),
    smtpSecure: boolean('smtp_secure').notNull().default(false),
    smtpUser: varchar('smtp_user', { length: 255 }),
    smtpPasswordEncrypted: text('smtp_password_encrypted'),
    fromAddress: varchar('from_address', { length: 255 }).notNull(),
    fromName: varchar('from_name', { length: 255 }).notNull(),
    replyTo: varchar('reply_to', { length: 255 }),
    clientPublicUrl: varchar('client_public_url', { length: 255 }),
    enabled: boolean('enabled').notNull().default(true),
    lastVerifiedAt: timestamp('last_verified_at', { withTimezone: true }),
    lastVerificationError: text('last_verification_error'),
    confirmedAt: timestamp('confirmed_at', { withTimezone: true }),
    confirmedByUserId: uuid('confirmed_by_user_id').references(() => users.id, {
        onDelete: 'set null',
    }),
    createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
    updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
    updatedByUserId: uuid('updated_by_user_id').references(() => users.id, {
        onDelete: 'set null',
    }),
});

export type MailSettingsSelect = typeof mailSettings.$inferSelect;
export type MailSettingsInsert = typeof mailSettings.$inferInsert;
