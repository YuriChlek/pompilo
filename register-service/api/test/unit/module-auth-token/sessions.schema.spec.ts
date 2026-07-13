import {
    sessions,
    isSessionActive,
    isSessionReusable,
    SessionSelect,
} from '@/module-auth-token/schemas/sessions.schema';
import { getTableConfig } from 'drizzle-orm/pg-core';

describe('SessionsSchema', () => {
    it('should have the correct table name', () => {
        const config = getTableConfig(sessions);
        expect(config.name).toBe('sessions');
    });

    it('should have all required columns', () => {
        const config = getTableConfig(sessions);
        const columnNames = config.columns.map(c => c.name);

        expect(columnNames).toContain('session_id');
        expect(columnNames).toContain('user_id');
        expect(columnNames).toContain('realm');
        expect(columnNames).toContain('known_device_id');
        expect(columnNames).toContain('device_id');
        expect(columnNames).toContain('ip_address');
        expect(columnNames).toContain('user_agent');
        expect(columnNames).toContain('created_at');
        expect(columnNames).toContain('updated_at');
        expect(columnNames).toContain('last_seen_at');
        expect(columnNames).toContain('expires_at');
        expect(columnNames).toContain('revoked_at');
        expect(columnNames).toContain('last_country');
        expect(columnNames).toContain('last_region');
        expect(columnNames).toContain('last_city');
        expect(columnNames).toContain('risk_score');
        expect(columnNames).toContain('risk_reason');
    });

    it('should have correct foreign keys referencing users and known_devices', () => {
        const config = getTableConfig(sessions);
        expect(config.foreignKeys.length).toBe(2);

        const userFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'user_id'),
        );
        expect(userFk).toBeDefined();
        expect(userFk?.onDelete).toBe('cascade');

        const deviceFk = config.foreignKeys.find(fk =>
            fk.reference().columns.some(c => c.name === 'known_device_id'),
        );
        expect(deviceFk).toBeDefined();
        expect(deviceFk?.onDelete).toBe('restrict');
    });

    it('should have correct unique and btree indexes', () => {
        const config = getTableConfig(sessions);
        const indexes = config.indexes;
        const indexNames = indexes.map(idx => idx.config.name);

        expect(indexNames).toContain('sessions_user_active_idx');
        expect(indexNames).toContain('sessions_last_seen_idx');
        expect(indexNames).toContain('sessions_expiry_idx');
        expect(indexNames).toContain('sessions_user_realm_device_active_unique');

        const uniqueIdx = indexes.find(
            idx => idx.config.name === 'sessions_user_realm_device_active_unique',
        );
        expect(uniqueIdx).toBeDefined();
        expect(uniqueIdx?.config.unique).toBe(true);

        const activeIdx = indexes.find(idx => idx.config.name === 'sessions_user_active_idx');
        expect(activeIdx).toBeDefined();
        expect(activeIdx?.config.unique).toBeFalsy();
    });

    it('should restrict realm and risk_score values at the database level', () => {
        const config = getTableConfig(sessions);
        const checkNames = config.checks.map(check => check.name);
        expect(checkNames).toContain('sessions_realm_check');
        expect(checkNames).toContain('sessions_risk_score_check');
    });

    describe('predicates', () => {
        const mockSession = (overrides: Partial<SessionSelect> = {}): SessionSelect => ({
            id: 'session-id',
            userId: 'user-id',
            realm: 'customer',
            knownDeviceId: 'known-device-id',
            deviceId: 'device-id',
            ipAddress: null,
            userAgent: null,
            createdAt: new Date(),
            updatedAt: new Date(),
            lastSeenAt: new Date(),
            expiresAt: overrides.expiresAt ?? new Date(Date.now() + 600_000),
            revokedAt: overrides.revokedAt ?? null,
            lastCountry: null,
            lastRegion: null,
            lastCity: null,
            riskScore: 0,
            riskReason: null,
        });

        it('isSessionActive should return true for unrevoked non-expired session', () => {
            const session = mockSession();
            expect(isSessionActive(session)).toBe(true);
        });

        it('isSessionActive should return false for revoked session', () => {
            const session = mockSession({ revokedAt: new Date() });
            expect(isSessionActive(session)).toBe(false);
        });

        it('isSessionActive should return false for expired session', () => {
            const session = mockSession({ expiresAt: new Date(Date.now() - 600_000) });
            expect(isSessionActive(session)).toBe(false);
        });

        it('isSessionReusable should return true for unrevoked session even if expired', () => {
            const session = mockSession({ expiresAt: new Date(Date.now() - 600_000) });
            expect(isSessionReusable(session)).toBe(true);
        });

        it('isSessionReusable should return false for revoked session', () => {
            const session = mockSession({ revokedAt: new Date() });
            expect(isSessionReusable(session)).toBe(false);
        });
    });
});
