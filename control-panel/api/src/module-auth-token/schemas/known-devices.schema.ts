import {
    check,
    index,
    pgTable,
    text,
    timestamp,
    uniqueIndex,
    uuid,
    varchar,
} from 'drizzle-orm/pg-core';
import { sql } from 'drizzle-orm';
import { users } from '@/module-user/schemas';

export const knownDevices = pgTable(
    'known_devices',
    {
        id: uuid('known_device_id').primaryKey().defaultRandom(),
        userId: uuid('user_id')
            .notNull()
            .references(() => users.id, { onDelete: 'no action' }),
        realm: varchar('realm', { length: 255 }).notNull(),
        deviceId: uuid('device_id').notNull(),
        trustedAt: timestamp('trusted_at', { withTimezone: true }),
        trustExpiresAt: timestamp('trust_expires_at', { withTimezone: true }),
        revokedAt: timestamp('revoked_at', { withTimezone: true }),
        firstSeenAt: timestamp('first_seen_at', { withTimezone: true }).notNull().defaultNow(),
        lastSeenAt: timestamp('last_seen_at', { withTimezone: true }).notNull().defaultNow(),
        lastIpAddress: varchar('last_ip_address', { length: 45 }),
        lastCountry: varchar('last_country', { length: 2 }),
        lastRegion: varchar('last_region', { length: 128 }),
        lastCity: varchar('last_city', { length: 128 }),
        lastUserAgent: text('last_user_agent'),
        createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
        updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
    },
    table => ({
        knownDevicesRealmCheck: check(
            'known_devices_realm_check',
            sql`${table.realm} in ('customer', 'admin')`,
        ),
        knownDevicesUserRealmDeviceUnique: uniqueIndex('known_devices_user_realm_device_unique')
            .on(table.userId, table.realm, table.deviceId)
            .where(sql`revoked_at is null`),
        knownDevicesUserLastSeenIdx: index('known_devices_user_last_seen_idx')
            .on(table.userId, table.realm, table.lastSeenAt)
            .where(sql`revoked_at is null`),
    }),
);

export type KnownDeviceSelect = typeof knownDevices.$inferSelect;
export type KnownDeviceInsert = typeof knownDevices.$inferInsert;

export const isDeviceTrusted = (device: KnownDeviceSelect, now = new Date()): boolean => {
    if (device.revokedAt !== null) {
        return false;
    }
    if (device.trustedAt === null) {
        return false;
    }
    if (device.trustExpiresAt !== null && device.trustExpiresAt.getTime() <= now.getTime()) {
        return false;
    }
    return true;
};
