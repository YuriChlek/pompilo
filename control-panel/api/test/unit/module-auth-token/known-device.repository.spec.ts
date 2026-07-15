import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import {
    KnownDeviceSelect,
    KnownDeviceInsert,
} from '@/module-auth-token/schemas/known-devices.schema';

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

describe('KnownDeviceRepository', () => {
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

    let repository: KnownDeviceRepository;

    beforeEach(() => {
        jest.clearAllMocks();
        repository = new KnownDeviceRepository(db as never);
    });

    it('should save a known device', async () => {
        const insertData: KnownDeviceInsert = {
            userId: 'user-uuid',
            realm: 'customer',
            deviceId: 'device-uuid',
        };
        const createdRow = buildMockDevice(insertData);

        insertReturning.mockResolvedValue([createdRow]);
        insertValues.mockReturnValue({ returning: insertReturning });
        insert.mockReturnValue({ values: insertValues });

        const result = await repository.save(insertData);

        expect(insert).toHaveBeenCalled();
        expect(insertValues).toHaveBeenCalledWith(insertData);
        expect(result).toEqual(createdRow);
    });

    it('should find by id', async () => {
        const row = buildMockDevice();

        selectLimit.mockResolvedValue([row]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findById('known-device-uuid');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should find active by device identity parameters', async () => {
        const row = buildMockDevice();

        selectLimit.mockResolvedValue([row]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findActiveByDevice('user-uuid', 'customer', 'device-uuid');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should update known device data and return updated row', async () => {
        const row = buildMockDevice({ lastIpAddress: '8.8.8.8' });

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.update('known-device-uuid', { lastIpAddress: '8.8.8.8' });

        expect(update).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should revoke known device by setting revokedAt', async () => {
        const row = buildMockDevice({ revokedAt: new Date() });

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.revoke('known-device-uuid');

        expect(update).toHaveBeenCalled();
        expect(result).toBe(true);
    });

    it('should list active known devices for a user and realm', async () => {
        const devices = [buildMockDevice({ id: '1' }), buildMockDevice({ id: '2' })];

        selectWhere.mockResolvedValue(devices);
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.listActive('user-uuid', 'customer');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(devices);
    });
});
