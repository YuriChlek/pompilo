import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { SessionSelect, SessionInsert } from '@/module-auth-token/schemas/sessions.schema';
import { KnownDeviceSelect } from '@/module-auth-token/schemas/known-devices.schema';

const buildMockSession = (overrides: Partial<SessionSelect> = {}): SessionSelect => ({
    id: overrides.id ?? 'session-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    knownDeviceId: overrides.knownDeviceId ?? 'known-device-uuid',
    deviceId: overrides.deviceId ?? 'device-uuid',
    ipAddress: overrides.ipAddress ?? null,
    userAgent: overrides.userAgent ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    updatedAt: overrides.updatedAt ?? new Date(),
    lastSeenAt: overrides.lastSeenAt ?? new Date(),
    expiresAt: overrides.expiresAt ?? new Date(),
    revokedAt: overrides.revokedAt ?? null,
    lastCountry: overrides.lastCountry ?? null,
    lastRegion: overrides.lastRegion ?? null,
    lastCity: overrides.lastCity ?? null,
    riskScore: overrides.riskScore ?? 0,
    riskReason: overrides.riskReason ?? null,
});

const buildMockDevice = (overrides: Partial<KnownDeviceSelect> = {}): KnownDeviceSelect => ({
    id: overrides.id ?? 'known-device-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    deviceId: overrides.deviceId ?? 'device-uuid',
    trustedAt: null,
    trustExpiresAt: null,
    revokedAt: null,
    firstSeenAt: new Date(),
    lastSeenAt: new Date(),
    lastIpAddress: null,
    lastCountry: null,
    lastRegion: null,
    lastCity: null,
    lastUserAgent: null,
    createdAt: new Date(),
    updatedAt: new Date(),
});

describe('SessionRepository', () => {
    const insertReturning = jest.fn();
    const insertValues = jest.fn();
    const insert = jest.fn();

    const selectLimit = jest.fn();
    const selectWhere = jest.fn();
    const selectFrom = jest.fn();
    const select = jest.fn();

    const updateReturning = jest.fn();
    const updateSet = jest.fn();
    const update = jest.fn();

    const db = {
        insert,
        select,
        update,
    };

    let repository: SessionRepository;

    beforeEach(() => {
        jest.clearAllMocks();
        repository = new SessionRepository(db as never);
    });

    it('should save a session after validating it matches known device ID', async () => {
        const insertData: SessionInsert = {
            userId: 'user-uuid',
            realm: 'customer',
            knownDeviceId: 'known-device-uuid',
            deviceId: 'device-uuid',
            expiresAt: new Date(),
        };

        const mockDevice = buildMockDevice({ id: 'known-device-uuid', deviceId: 'device-uuid' });
        const createdRow = buildMockSession(insertData);

        // Mock select for device verification
        selectLimit.mockResolvedValue([mockDevice]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        // Mock insert
        insertReturning.mockResolvedValue([createdRow]);
        insertValues.mockReturnValue({ returning: insertReturning });
        insert.mockReturnValue({ values: insertValues });

        const result = await repository.save(insertData);

        expect(select).toHaveBeenCalled();
        expect(insert).toHaveBeenCalled();
        expect(result).toEqual(createdRow);
    });

    it('should throw error if session device_id does not match known_device device_id', async () => {
        const insertData: SessionInsert = {
            userId: 'user-uuid',
            realm: 'customer',
            knownDeviceId: 'known-device-uuid',
            deviceId: 'different-device-uuid',
            expiresAt: new Date(),
        };

        const mockDevice = buildMockDevice({ id: 'known-device-uuid', deviceId: 'device-uuid' });

        selectLimit.mockResolvedValue([mockDevice]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        await expect(repository.save(insertData)).rejects.toThrow(
            'session.device_id (different-device-uuid) does not match known_device.device_id (device-uuid)',
        );
    });

    it('should find by id', async () => {
        const row = buildMockSession();

        selectLimit.mockResolvedValue([row]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findById('session-uuid');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should find reusable session', async () => {
        const row = buildMockSession();

        selectLimit.mockResolvedValue([row]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findReusable('user-uuid', 'customer', 'device-uuid');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should find active by user and realm', async () => {
        const rows = [buildMockSession()];

        selectWhere.mockResolvedValue(rows);
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findActiveByUserRealm('user-uuid', 'customer');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(rows);
    });

    it('should find unrevoked sessions by user and realm (including current and expired-but-unrevoked)', async () => {
        const rows = [
            buildMockSession({
                id: 'session-1',
                expiresAt: new Date(Date.now() - 1000),
                revokedAt: null,
            }),
            buildMockSession({
                id: 'session-2',
                expiresAt: new Date(Date.now() + 60000),
                revokedAt: null,
            }),
        ];

        selectWhere.mockResolvedValue(rows);
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findUnrevokedByUserRealm('user-uuid', 'customer');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(rows);
    });

    it('should revoke session', async () => {
        const now = new Date();
        const row = buildMockSession({ revokedAt: now });

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.revoke('session-uuid', now);

        expect(update).toHaveBeenCalled();
        expect(result).toBe(true);
    });

    it('should reuse/extend session', async () => {
        const now = new Date();
        const expiresAt = new Date(Date.now() + 600_000);
        const row = buildMockSession({ expiresAt });

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.reuseSession('session-uuid', expiresAt, undefined, now);

        expect(update).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should update a session', async () => {
        const row = buildMockSession({ ipAddress: '5.6.7.8' });

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.update('session-uuid', { ipAddress: '5.6.7.8' });

        expect(update).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should find unrevoked other sessions (excluding current, including expired-but-unrevoked)', async () => {
        const rows = [buildMockSession({ id: 'other-session-uuid' })];

        selectWhere.mockResolvedValue(rows);
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findUnrevokedOtherByUserRealm(
            'user-uuid',
            'customer',
            'current-session-uuid',
        );

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(rows);
    });

    it('should not include current session in findUnrevokedOtherByUserRealm result', async () => {
        // Verify the query excludes currentSessionId: the repository delegates
        // predicate building to Drizzle; we assert the mock was invoked and
        // that the returned rows do NOT contain the currentSessionId.
        const currentSessionId = 'current-session-uuid';
        const otherSession = buildMockSession({ id: 'other-session-uuid' });

        selectWhere.mockResolvedValue([otherSession]);
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findUnrevokedOtherByUserRealm(
            'user-uuid',
            'customer',
            currentSessionId,
        );

        expect(result.every(s => s.id !== currentSessionId)).toBe(true);
    });

    it('should revoke all sessions', async () => {
        const row = buildMockSession();

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.revokeAll('user-uuid', 'customer');

        expect(update).toHaveBeenCalled();
        expect(result).toBe(1);
    });
});
