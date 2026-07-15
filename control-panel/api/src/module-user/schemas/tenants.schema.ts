import { pgEnum, pgTable, primaryKey, uuid, varchar, timestamp } from 'drizzle-orm/pg-core';
import { users } from './users.schema';

export const tenantStatusEnum = pgEnum('tenant_status_enum', ['ACTIVE', 'SUSPENDED']);
export const tenantRoleEnum = pgEnum('tenant_role_enum', ['OWNER', 'ADMIN', 'MEMBER']);

export const tenants = pgTable('tenants', {
    id: uuid('tenant_id').primaryKey().defaultRandom(),
    name: varchar('name', { length: 255 }).notNull(),
    status: tenantStatusEnum('status').notNull().default('ACTIVE'),
    createdAt: timestamp('created_at').notNull().defaultNow(),
    updatedAt: timestamp('updated_at').notNull().defaultNow(),
});

export const memberships = pgTable(
    'memberships',
    {
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        tenantId: uuid('tenant_id')
            .notNull()
            .references(() => tenants.id, { onDelete: 'cascade' }),
        role: tenantRoleEnum('role').notNull().default('MEMBER'),
        createdAt: timestamp('created_at').notNull().defaultNow(),
        updatedAt: timestamp('updated_at').notNull().defaultNow(),
    },
    table => ({
        membershipsPk: primaryKey({ columns: [table.userId, table.tenantId] }),
    }),
);

export type TenantSelect = typeof tenants.$inferSelect;
export type TenantInsert = typeof tenants.$inferInsert;
export type MembershipSelect = typeof memberships.$inferSelect;
export type MembershipInsert = typeof memberships.$inferInsert;
