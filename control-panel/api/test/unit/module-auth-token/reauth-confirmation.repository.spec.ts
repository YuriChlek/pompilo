import { ReauthConfirmationRepository } from '@/module-auth-token/repository/reauth-confirmation.repository';
import {
    ReauthConfirmationSelect,
    ReauthConfirmationInsert,
} from '@/module-auth-token/schemas/reauth-confirmations.schema';

const buildMockConfirmation = (
    overrides: Partial<ReauthConfirmationSelect> = {},
): ReauthConfirmationSelect => ({
    id: overrides.id ?? 'confirmation-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    sessionId: overrides.sessionId ?? 'session-uuid',
    actionScope: overrides.actionScope ?? 'email_change',
    confirmationTokenHash: overrides.confirmationTokenHash ?? 'token-hash',
    expiresAt: overrides.expiresAt ?? new Date(),
    consumedAt: overrides.consumedAt ?? null,
    createdAt: overrides.createdAt ?? new Date(),
});

describe('ReauthConfirmationRepository', () => {
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

    let repository: ReauthConfirmationRepository;

    beforeEach(() => {
        jest.clearAllMocks();
        repository = new ReauthConfirmationRepository(db as never);
    });

    it('should save a reauth confirmation', async () => {
        const insertData: ReauthConfirmationInsert = {
            userId: 'user-uuid',
            realm: 'customer',
            sessionId: 'session-uuid',
            actionScope: 'email_change',
            confirmationTokenHash: 'token-hash',
            expiresAt: new Date(),
        };
        const createdRow = buildMockConfirmation(insertData);

        insertReturning.mockResolvedValue([createdRow]);
        insertValues.mockReturnValue({ returning: insertReturning });
        insert.mockReturnValue({ values: insertValues });

        const result = await repository.save(insertData);

        expect(insert).toHaveBeenCalled();
        expect(insertValues).toHaveBeenCalledWith(insertData);
        expect(result).toEqual(createdRow);
    });

    it('should find by id', async () => {
        const row = buildMockConfirmation();

        selectLimit.mockResolvedValue([row]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findById('confirmation-uuid');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should find by token hash', async () => {
        const row = buildMockConfirmation();

        selectLimit.mockResolvedValue([row]);
        selectWhere.mockReturnValue({ limit: selectLimit });
        selectFrom.mockReturnValue({ where: selectWhere });
        select.mockReturnValue({ from: selectFrom });

        const result = await repository.findByTokenHash('token-hash');

        expect(select).toHaveBeenCalled();
        expect(result).toEqual(row);
    });

    it('should consume confirmation by id', async () => {
        const now = new Date();
        const row = buildMockConfirmation({ consumedAt: now });

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.consume('confirmation-uuid', now);

        expect(update).toHaveBeenCalled();
        expect(result).toBe(true);
    });

    it('should consume confirmation atomically by token hash', async () => {
        const now = new Date();
        const row = buildMockConfirmation({ consumedAt: now });

        updateReturning.mockResolvedValue([row]);
        updateSet.mockReturnValue({
            where: jest.fn().mockReturnValue({ returning: updateReturning }),
        });
        update.mockReturnValue({ set: updateSet });

        const result = await repository.consumeByTokenHash(
            'token-hash',
            'user-uuid',
            'customer',
            'session-uuid',
            'email_change',
            now,
        );

        expect(update).toHaveBeenCalled();
        expect(result).toBe(true);
    });
});
