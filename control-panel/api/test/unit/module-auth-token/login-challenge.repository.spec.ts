import { LoginChallengeRepository } from '@/module-auth-token/repository/login-challenge.repository';
import {
    LoginChallengeInsert,
    LoginChallengeSelect,
} from '@/module-auth-token/schemas/login-challenges.schema';

const buildMockChallenge = (
    overrides: Partial<LoginChallengeSelect> = {},
): LoginChallengeSelect => ({
    id: overrides.id ?? 'challenge-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    knownDeviceId: overrides.knownDeviceId ?? 'device-uuid',
    deviceId: overrides.deviceId ?? 'device-uuid-123',
    challengeType: overrides.challengeType ?? 'email_code',
    checkpointTokenHash: overrides.checkpointTokenHash ?? 'token-hash',
    codeHash: overrides.codeHash ?? 'code-hash',
    attemptCount: overrides.attemptCount ?? 0,
    maxAttempts: overrides.maxAttempts ?? 5,
    expiresAt: overrides.expiresAt ?? new Date(Date.now() + 600000),
    approvedAt: overrides.approvedAt ?? null,
    consumedAt: overrides.consumedAt ?? null,
    failedAt: overrides.failedAt ?? null,
    expiredAt: overrides.expiredAt ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    ipAddress: overrides.ipAddress ?? null,
    country: overrides.country ?? null,
    region: overrides.region ?? null,
    city: overrides.city ?? null,
    userAgent: overrides.userAgent ?? null,
    riskScore: overrides.riskScore ?? 0,
    riskReason: overrides.riskReason ?? null,
});

describe('LoginChallengeRepository', () => {
    const updateReturning = jest.fn();
    const updateSet = jest.fn();
    const update = jest.fn();

    const insertReturning = jest.fn();
    const insertValues = jest.fn();
    const insert = jest.fn();

    const selectLimit = jest.fn();
    const selectOrderBy = jest.fn();
    const selectWhere = jest.fn();
    const selectFrom = jest.fn();
    const select = jest.fn();

    const execute = jest.fn();

    // Mock Drizzle Transaction
    const transaction = jest.fn();

    const db = {
        transaction,
        update,
        insert,
        select,
        execute,
    };

    let repository: LoginChallengeRepository;

    beforeEach(() => {
        jest.clearAllMocks();
        repository = new LoginChallengeRepository(db as never);
    });

    describe('createSafe', () => {
        const insertData: LoginChallengeInsert = {
            userId: 'user-uuid',
            realm: 'customer',
            deviceId: 'device-uuid-123',
            checkpointTokenHash: 'token-hash',
            codeHash: 'code-hash',
            expiresAt: new Date(Date.now() + 600000),
        };

        it('should mark old expired challenges and insert a new one successfully', async () => {
            const createdRow = buildMockChallenge(insertData);

            // Mock transaction callback behavior
            const mockTx = {
                update,
                insert,
                execute,
            };
            transaction.mockImplementation(async (cb: (tx: unknown) => Promise<unknown>) => {
                return await cb(mockTx);
            });
            execute.mockResolvedValue({ rows: [] });

            // Mock update behavior inside transaction
            updateSet.mockReturnValue({ where: jest.fn().mockResolvedValue({ rowCount: 0 }) });
            update.mockReturnValue({ set: updateSet });

            // Mock insert behavior inside transaction
            insertReturning.mockResolvedValue([createdRow]);
            insertValues.mockReturnValue({ returning: insertReturning });
            insert.mockReturnValue({ values: insertValues });

            const result = await repository.createSafe(insertData);

            expect(transaction).toHaveBeenCalled();
            expect(execute).toHaveBeenCalled();
            expect(update).toHaveBeenCalled();
            expect(insert).toHaveBeenCalled();
            expect(insertValues).toHaveBeenCalledWith(insertData);
            expect(result).toEqual(createdRow);
        });

        it('should propagate any other error that is not a unique violation', async () => {
            const dbError = new Error('Database connection failed');
            transaction.mockRejectedValue(dbError);

            await expect(repository.createSafe(insertData)).rejects.toThrow(
                'Database connection failed',
            );
        });
    });

    describe('findById', () => {
        it('should query and return target challenge by id', async () => {
            const row = buildMockChallenge();

            selectLimit.mockResolvedValue([row]);
            selectWhere.mockReturnValue({ limit: selectLimit });
            selectFrom.mockReturnValue({ where: selectWhere });
            select.mockReturnValue({ from: selectFrom });

            const result = await repository.findById('challenge-uuid');

            expect(select).toHaveBeenCalled();
            expect(result).toEqual(row);
        });
    });

    describe('findLatestByUserRealmDevice', () => {
        it('should query and return latest challenge for user realm and device', async () => {
            const row = buildMockChallenge();

            selectLimit.mockResolvedValue([row]);
            selectOrderBy.mockReturnValue({ limit: selectLimit });
            selectWhere.mockReturnValue({ orderBy: selectOrderBy });
            selectFrom.mockReturnValue({ where: selectWhere });
            select.mockReturnValue({ from: selectFrom });

            const result = await repository.findLatestByUserRealmDevice(
                'user-uuid',
                'customer',
                'device-uuid-123',
            );

            expect(select).toHaveBeenCalled();
            expect(selectWhere).toHaveBeenCalled();
            expect(selectOrderBy).toHaveBeenCalled();
            expect(selectLimit).toHaveBeenCalledWith(1);
            expect(result).toEqual(row);
        });
    });

    describe('consume', () => {
        it('should set consumedAt timestamp on active row', async () => {
            const row = buildMockChallenge({ consumedAt: new Date() });

            updateReturning.mockResolvedValue([row]);
            updateSet.mockReturnValue({
                where: jest.fn().mockReturnValue({ returning: updateReturning }),
            });
            update.mockReturnValue({ set: updateSet });

            const result = await repository.consume('challenge-uuid');

            expect(update).toHaveBeenCalled();
            expect(result).toBe(true);
        });
    });

    describe('approveAndConsume', () => {
        it('should set approvedAt and consumedAt timestamps on active row', async () => {
            const row = buildMockChallenge({ approvedAt: new Date(), consumedAt: new Date() });

            updateReturning.mockResolvedValue([row]);
            updateSet.mockReturnValue({
                where: jest.fn().mockReturnValue({ returning: updateReturning }),
            });
            update.mockReturnValue({ set: updateSet });

            const result = await repository.approveAndConsume('challenge-uuid');

            expect(update).toHaveBeenCalled();
            expect(result).toBe(true);
        });
    });

    describe('fail', () => {
        it('should set failedAt timestamp on active row', async () => {
            const row = buildMockChallenge({ failedAt: new Date() });

            updateReturning.mockResolvedValue([row]);
            updateSet.mockReturnValue({
                where: jest.fn().mockReturnValue({ returning: updateReturning }),
            });
            update.mockReturnValue({ set: updateSet });

            const result = await repository.fail('challenge-uuid');

            expect(update).toHaveBeenCalled();
            expect(result).toBe(true);
        });
    });

    describe('incrementAttempts', () => {
        it('should increment attemptCount and return the updated row', async () => {
            const row = buildMockChallenge({ attemptCount: 1 });

            updateReturning.mockResolvedValue([row]);
            updateSet.mockReturnValue({
                where: jest.fn().mockReturnValue({ returning: updateReturning }),
            });
            update.mockReturnValue({ set: updateSet });

            const result = await repository.incrementAttempts('challenge-uuid');

            expect(update).toHaveBeenCalled();
            expect(result).toEqual(row);
        });
    });

    describe('expire', () => {
        it('should materialize expiredAt for a stale active challenge', async () => {
            const row = buildMockChallenge({ expiredAt: new Date() });

            updateReturning.mockResolvedValue([row]);
            updateSet.mockReturnValue({
                where: jest.fn().mockReturnValue({ returning: updateReturning }),
            });
            update.mockReturnValue({ set: updateSet });

            const result = await repository.expire('challenge-uuid');

            expect(update).toHaveBeenCalled();
            expect(result).toBe(true);
        });
    });
});
