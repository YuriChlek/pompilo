import {
    knownDevices,
    isDeviceTrusted,
    KnownDeviceSelect,
} from '@/module-auth-token/schemas/known-devices.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

const buildMockDevice = (overrides: Partial<KnownDeviceSelect> = {}): KnownDeviceSelect => ({
    id: overrides.id ?? 'known-device-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    deviceId: overrides.deviceId ?? 'device-uuid',
    trustedAt: overrides.trustedAt ?? null,
    trustExpiresAt: overrides.trustExpiresAt ?? null,
    revokedAt: overrides.revokedAt ?? null,
    firstSeenAt: overrides.firstSeenAt ?? new Date(),
    lastSeenAt: overrides.lastSeenAt ?? new Date(),
    lastIpAddress: overrides.lastIpAddress ?? null,
    lastCountry: overrides.lastCountry ?? null,
    lastRegion: overrides.lastRegion ?? null,
    lastCity: overrides.lastCity ?? null,
    lastUserAgent: overrides.lastUserAgent ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    updatedAt: overrides.updatedAt ?? new Date(),
});

describe('KnownDevicesSchema', () => {
    it('should have the correct table name', () => {
        const config = getTableConfig(knownDevices);
        expect(config.name).toBe('known_devices');
    });

    it('should have all required columns', () => {
        const config = getTableConfig(knownDevices);
        const columnNames = config.columns.map(c => c.name);

        expect(columnNames).toContain('known_device_id');
        expect(columnNames).toContain('user_id');
        expect(columnNames).toContain('realm');
        expect(columnNames).toContain('device_id');
        expect(columnNames).toContain('trusted_at');
        expect(columnNames).toContain('trust_expires_at');
        expect(columnNames).toContain('revoked_at');
        expect(columnNames).toContain('first_seen_at');
        expect(columnNames).toContain('last_seen_at');
        expect(columnNames).toContain('last_ip_address');
        expect(columnNames).toContain('last_country');
        expect(columnNames).toContain('last_region');
        expect(columnNames).toContain('last_city');
        expect(columnNames).toContain('last_user_agent');
        expect(columnNames).toContain('created_at');
        expect(columnNames).toContain('updated_at');
    });

    it('should have correct foreign key referencing users', () => {
        const config = getTableConfig(knownDevices);
        expect(config.foreignKeys.length).toBeGreaterThan(0);

        const userFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'user_id'),
        );
        expect(userFk).toBeDefined();
        expect(userFk?.onDelete).toBe('no action');
    });

    it('should have correct unique and btree indexes', () => {
        const config = getTableConfig(knownDevices);
        const indexes = config.indexes;
        const indexNames = indexes.map(idx => idx.config.name);

        expect(indexNames).toContain('known_devices_user_realm_device_unique');
        expect(indexNames).toContain('known_devices_user_last_seen_idx');

        const uniqueIdx = indexes.find(
            idx => idx.config.name === 'known_devices_user_realm_device_unique',
        );
        expect(uniqueIdx).toBeDefined();
        expect(uniqueIdx?.config.unique).toBe(true);

        const lastSeenIdx = indexes.find(
            idx => idx.config.name === 'known_devices_user_last_seen_idx',
        );
        expect(lastSeenIdx).toBeDefined();
        expect(lastSeenIdx?.config.unique).toBeFalsy();
    });

    it('should restrict realm values at the database level', () => {
        const config = getTableConfig(knownDevices);
        expect(config.checks.map(check => check.name)).toContain('known_devices_realm_check');
    });

    describe('isDeviceTrusted', () => {
        it('should return false if revoked', () => {
            const device = buildMockDevice({
                trustedAt: new Date(),
                revokedAt: new Date(),
            });
            expect(isDeviceTrusted(device)).toBe(false);
        });

        it('should return false if trustedAt is null', () => {
            const device = buildMockDevice({
                trustedAt: null,
            });
            expect(isDeviceTrusted(device)).toBe(false);
        });

        it('should return true if trustedAt is set and trustExpiresAt is null', () => {
            const device = buildMockDevice({
                trustedAt: new Date(),
                trustExpiresAt: null,
            });
            expect(isDeviceTrusted(device)).toBe(true);
        });

        it('should return true if trustExpiresAt is in the future', () => {
            const now = new Date();
            const future = new Date(now.getTime() + 10000);
            const device = buildMockDevice({
                trustedAt: now,
                trustExpiresAt: future,
            });
            expect(isDeviceTrusted(device, now)).toBe(true);
        });

        it('should return false if trustExpiresAt is in the past', () => {
            const now = new Date();
            const past = new Date(now.getTime() - 10000);
            const device = buildMockDevice({
                trustedAt: now,
                trustExpiresAt: past,
            });
            expect(isDeviceTrusted(device, now)).toBe(false);
        });
    });
});
