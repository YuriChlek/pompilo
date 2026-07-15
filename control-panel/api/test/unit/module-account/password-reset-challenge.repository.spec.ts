import { Test, TestingModule } from '@nestjs/testing';
import { PasswordResetChallengeRepository } from '@/module-account/repository/password-reset-challenge.repository';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import { PasswordResetChallengeInsert } from '@/module-account/schemas';

describe('PasswordResetChallengeRepository', () => {
    let repository: PasswordResetChallengeRepository;
    let mockDb: {
        insert: jest.Mock;
        values: jest.Mock;
        returning: jest.Mock;
        select: jest.Mock;
        from: jest.Mock;
        where: jest.Mock;
        limit: jest.Mock;
        update: jest.Mock;
        set: jest.Mock;
        delete: jest.Mock;
    };

    beforeEach(async () => {
        mockDb = {
            insert: jest.fn().mockReturnThis(),
            values: jest.fn().mockReturnThis(),
            returning: jest.fn().mockResolvedValue([{ id: 'test-id' }]),
            select: jest.fn().mockReturnThis(),
            from: jest.fn().mockReturnThis(),
            where: jest.fn().mockReturnThis(),
            limit: jest.fn().mockReturnThis(),
            update: jest.fn().mockReturnThis(),
            set: jest.fn().mockReturnThis(),
            delete: jest.fn().mockReturnThis(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                PasswordResetChallengeRepository,
                { provide: DRIZZLE_PROVIDER, useValue: mockDb },
            ],
        }).compile();

        repository = module.get<PasswordResetChallengeRepository>(PasswordResetChallengeRepository);
    });

    it('should create a challenge', async () => {
        const data: PasswordResetChallengeInsert = {
            userId: '00000000-0000-0000-0000-000000000001',
            selector: 'sel-1',
            verifierDigest: 'digest-1',
            expiresAt: new Date(),
        };

        await repository.create(data);

        expect(mockDb.insert).toHaveBeenCalled();
        expect(mockDb.values).toHaveBeenCalledWith(data);
    });

    it('should find by selector', async () => {
        const selector = 'sel-1';
        mockDb.limit.mockResolvedValue([{ id: 'test-id', selector }]);

        const result = await repository.findBySelector(selector);

        expect(mockDb.select).toHaveBeenCalled();
        expect(mockDb.where).toHaveBeenCalled();
        expect(result).toEqual({ id: 'test-id', selector });
    });

    it('should find active by selector', async () => {
        const selector = 'sel-1';
        mockDb.limit.mockResolvedValue([{ id: 'test-id', selector }]);

        const result = await repository.findActiveBySelector(selector);

        expect(mockDb.select).toHaveBeenCalled();
        expect(mockDb.where).toHaveBeenCalled();
        expect(result).toEqual({ id: 'test-id', selector });
    });

    it('should consume a challenge', async () => {
        const id = 'ch-1';
        mockDb.returning.mockResolvedValue([{ id }]);

        const result = await repository.consume(id);

        expect(mockDb.update).toHaveBeenCalled();
        expect(mockDb.set).toHaveBeenCalledWith({ usedAt: expect.any(Date) as Date });
        expect(mockDb.where).toHaveBeenCalled();
        expect(result).toBe(true);
    });

    it('should invalidate user challenges', async () => {
        const userId = '00000000-0000-0000-0000-000000000001';

        await repository.invalidateUserChallenges(userId);

        expect(mockDb.update).toHaveBeenCalled();
        expect(mockDb.set).toHaveBeenCalledWith({ invalidatedAt: expect.any(Date) as Date });
        expect(mockDb.where).toHaveBeenCalled();
    });

    it('should update a challenge', async () => {
        const id = '00000000-0000-0000-0000-000000000001';
        const usedAt = new Date();
        const data: Partial<PasswordResetChallengeInsert> = { usedAt };

        await repository.update(id, data);

        expect(mockDb.update).toHaveBeenCalled();
        expect(mockDb.set).toHaveBeenCalledWith(data);
        expect(mockDb.where).toHaveBeenCalled();
    });

    it('should cleanup old challenges', async () => {
        const olderThan = new Date();
        mockDb.returning.mockResolvedValue([{ id: '1' }, { id: '2' }]);

        const count = await repository.cleanup(olderThan);

        expect(mockDb.delete).toHaveBeenCalled();
        expect(mockDb.where).toHaveBeenCalled();
        expect(count).toBe(2);
    });
});
