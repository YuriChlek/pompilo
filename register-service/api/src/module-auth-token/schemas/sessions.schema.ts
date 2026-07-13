import {
    check,
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
import { users } from '@/module-user/schemas';
import { knownDevices } from '@/module-auth-token/schemas/known-devices.schema';

export const sessions = pgTable(
    'sessions',
    {
        id: uuid('session_id').primaryKey().defaultRandom(),
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'cascade' }),
        realm: varchar('realm', { length: 255 }).notNull(),
        knownDeviceId: uuid('known_device_id')
            .notNull()
            .references(() => knownDevices.id, { onDelete: 'restrict' }),
        deviceId: uuid('device_id').notNull(),
        ipAddress: varchar('ip_address', { length: 45 }),
        userAgent: text('user_agent'),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
        updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
        lastSeenAt: timestamp('last_seen_at', { withTimezone: true }).notNull().defaultNow(),
        expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
        revokedAt: timestamp('revoked_at', { withTimezone: true }),
        lastCountry: varchar('last_country', { length: 2 }),
        lastRegion: varchar('last_region', { length: 128 }),
        lastCity: varchar('last_city', { length: 128 }),
        riskScore: integer('risk_score').notNull().default(0),
        riskReason: text('risk_reason'),
    },
    table => ({
        sessionsRealmCheck: check(
            'sessions_realm_check',
            sql`${table.realm} in ('customer', 'admin')`,
        ),
        sessionsRiskScoreCheck: check('sessions_risk_score_check', sql`${table.riskScore} >= 0`),
        sessionsUserActiveIdx: index('sessions_user_active_idx')
            .on(table.userId, table.realm)
            .where(sql`revoked_at is null`),
        sessionsLastSeenIdx: index('sessions_last_seen_idx')
            .on(table.userId, table.realm, table.lastSeenAt)
            .where(sql`revoked_at is null`),
        sessionsExpiryIdx: index('sessions_expiry_idx')
            .on(table.expiresAt)
            .where(sql`revoked_at is null`),
        sessionsUserRealmDeviceActiveUnique: uniqueIndex('sessions_user_realm_device_active_unique')
            .on(table.userId, table.realm, table.deviceId)
            .where(sql`revoked_at is null`),
    }),
);

export type SessionSelect = typeof sessions.$inferSelect;
export type SessionInsert = typeof sessions.$inferInsert;

export const isSessionActive = (session: SessionSelect, now = new Date()): boolean => {
    return session.revokedAt === null && session.expiresAt.getTime() > now.getTime();
};

export const isSessionReusable = (session: SessionSelect): boolean => {
    return session.revokedAt === null;
};
