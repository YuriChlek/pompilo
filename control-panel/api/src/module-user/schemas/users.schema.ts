import {
    boolean,
    pgEnum,
    pgTable,
    text,
    timestamp,
    uniqueIndex,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { UserRoles } from '@/module-auth/enums/auth.enums';

export const identityUserRoleEnum = pgEnum('identity_user_role_enum', [
    UserRoles.USER,
    UserRoles.PLATFORM_ADMIN,
    UserRoles.SUPER_ADMIN,
]);

export const accountStatusEnum = pgEnum('account_status_enum', [
    'ACTIVE',
    'DEACTIVATED',
    'PENDING_DELETION',
    'DELETED',
]);

export const users = pgTable(
    'users',
    {
        id: uuid('user_id').primaryKey().defaultRandom(),
        name: varchar('name', { length: 255 }).notNull(),
        email: varchar('email', { length: 255 }).notNull(),
        password: text('password').notNull(),
        role: identityUserRoleEnum('role').notNull().default(UserRoles.USER),
        isActive: boolean('is_active').notNull().default(true),
        accountStatus: accountStatusEnum('account_status').notNull().default('ACTIVE'),
        emailVerifiedAt: timestamp('email_verified_at', { withTimezone: true }),
        pendingEmailChange: varchar('pending_email_change', { length: 255 }),
        deletionScheduledAt: timestamp('deletion_scheduled_at', { withTimezone: true }),
        createdAt: timestamp('created_at').notNull().defaultNow(),
        updatedAt: timestamp('updated_at').notNull().defaultNow(),
    },
    table => ({
        usersNameUniqueIdx: uniqueIndex('users_name_unique_idx').on(table.name),
        usersEmailUniqueIdx: uniqueIndex('users_email_unique_idx').on(table.email),
        usersNormalizedEmailUniqueIdx: uniqueIndex('users_normalized_email_unique_idx').on(
            sql`lower(${table.email})`,
        ),
    }),
);

export type UserSelect = typeof users.$inferSelect;
export type UserInsert = typeof users.$inferInsert;
