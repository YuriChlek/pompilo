import { SecurityEventRepository } from '@/module-auth-token/repository/security-event.repository';
import { SecurityEventSelect } from '@/module-auth-token/schemas/security-events.schema';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import type { SecurityEventWriteInput } from '@/module-auth-token/interfaces/security-event.interfaces';

const buildSecurityEventRow = (
    overrides: Partial<SecurityEventSelect> = {},
): SecurityEventSelect => ({
    id: overrides.id ?? 'event-uuid-1',
    userId: overrides.userId ?? 'user-uuid-1',
    realm: overrides.realm ?? 'customer',
    sessionId: overrides.sessionId ?? 'session-uuid-1',
    knownDeviceId: overrides.knownDeviceId ?? 'device-uuid-1',
    eventType: overrides.eventType ?? SecurityEventType.LOGIN_SUCCESS,
    riskScore: overrides.riskScore ?? 0,
    riskReason: overrides.riskReason ?? null,
    ipAddress: overrides.ipAddress ?? '127.0.0.1',
    country: overrides.country ?? 'US',
    region: overrides.region ?? 'California',
    city: overrides.city ?? 'San Francisco',
    userAgent: overrides.userAgent ?? 'Mozilla/5.0',
    createdAt: overrides.createdAt ?? new Date('2026-06-22T19:22:59.000Z'),
    metadata: overrides.metadata ?? { os: 'linux' },
});

describe('SecurityEventRepository', () => {
    const insertReturning = jest.fn();
    const insertValues = jest.fn();
    const insert = jest.fn();
    const select = jest.fn();
    const from = jest.fn();
    const where = jest.fn();
    const limit = jest.fn();
    const orderBy = jest.fn();

    const db = {
        insert,
        select,
    };

    let repository: SecurityEventRepository;

    beforeEach(() => {
        jest.clearAllMocks();
        repository = new SecurityEventRepository(db as never);
    });

    it('should save a security event preserving optional identifiers and metadata', async () => {
        const insertData: SecurityEventWriteInput = {
            userId: 'user-uuid-1',
            realm: 'customer',
            sessionId: 'session-uuid-1',
            knownDeviceId: 'device-uuid-1',
            eventType: SecurityEventType.LOGIN_SUCCESS,
            riskScore: 0,
            riskReason: null,
            ipAddress: '127.0.0.1',
            country: 'US',
            region: 'California',
            city: 'San Francisco',
            userAgent: 'Mozilla/5.0',
            metadata: { os: 'linux' },
        };

        const createdRow = buildSecurityEventRow(insertData);

        insertReturning.mockResolvedValue([createdRow]);
        insertValues.mockReturnValue({ returning: insertReturning });
        insert.mockReturnValue({ values: insertValues });

        const result = await repository.save(insertData);

        expect(insert).toHaveBeenCalled();
        expect(insertValues).toHaveBeenCalledWith(insertData);
        expect(result).toEqual(createdRow);
    });

    it('should save a security event with pre-session (null session and device id) info correctly', async () => {
        const insertData: SecurityEventWriteInput = {
            userId: null,
            realm: 'customer',
            sessionId: null,
            knownDeviceId: null,
            eventType: SecurityEventType.LOGIN_FAILED,
            riskScore: 10,
            riskReason: 'Brute force attempt',
            ipAddress: '192.168.1.1',
            country: 'DE',
            region: 'Bavaria',
            city: 'Munich',
            userAgent: 'Mozilla/5.0',
            metadata: { failureReason: 'invalid_credentials' },
        };

        const createdRow = buildSecurityEventRow(insertData);

        insertReturning.mockResolvedValue([createdRow]);
        insertValues.mockReturnValue({ returning: insertReturning });
        insert.mockReturnValue({ values: insertValues });

        const result = await repository.save(insertData);

        expect(insertValues).toHaveBeenCalledWith(insertData);
        expect(result).toEqual(createdRow);
    });

    it('should find security event by id', async () => {
        const row = buildSecurityEventRow();

        limit.mockResolvedValue([row]);
        where.mockReturnValue({ limit });
        from.mockReturnValue({ where });
        select.mockReturnValue({ from });

        const result = await repository.findById('event-uuid-1');

        expect(select).toHaveBeenCalled();
        expect(from).toHaveBeenCalled();
        expect(where).toHaveBeenCalled();
        expect(limit).toHaveBeenCalledWith(1);
        expect(result).toEqual(row);
    });

    it('should find security events by user id', async () => {
        const rows = [buildSecurityEventRow()];

        orderBy.mockResolvedValue(rows);
        where.mockReturnValue({ orderBy });
        from.mockReturnValue({ where });
        select.mockReturnValue({ from });

        const result = await repository.findByUserId('user-uuid-1');

        expect(select).toHaveBeenCalled();
        expect(from).toHaveBeenCalled();
        expect(where).toHaveBeenCalled();
        expect(orderBy).toHaveBeenCalled();
        expect(result).toEqual(rows);
    });

    it('should reject an event that misses taxonomy-required metadata', async () => {
        await expect(
            repository.save({
                userId: null,
                realm: 'customer',
                sessionId: null,
                knownDeviceId: null,
                eventType: SecurityEventType.LOGIN_FAILED,
                ipAddress: '192.0.2.1',
                metadata: {},
            }),
        ).rejects.toThrow('login_failed requires metadata field "failureReason"');

        expect(insert).not.toHaveBeenCalled();
    });
});
