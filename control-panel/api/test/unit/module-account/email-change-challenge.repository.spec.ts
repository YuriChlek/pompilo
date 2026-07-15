import { Test, TestingModule } from '@nestjs/testing';
import { EmailChangeChallengeRepository } from '@/module-account/repository/email-change-challenge.repository';
import { DRIZZLE_PROVIDER } from '@/module-drizzle/providers/drizzle.provider';
import { EmailChangeChallengeInsert } from '@/module-account/schemas';

describe('EmailChangeChallengeRepository', () => {
    let repository: EmailChangeChallengeRepository;
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
                EmailChangeChallengeRepository,
                { provide: DRIZZLE_PROVIDER, useValue: mockDb },
            ],
        }).compile();

        repository = module.get<EmailChangeChallengeRepository>(EmailChangeChallengeRepository);
    });

    it('should create a challenge', async () => {
        const data: EmailChangeChallengeInsert = {
            userId: '00000000-0000-0000-0000-000000000001',
            newEmail: 'new@test.com',
            codeDigest: 'digest-1',
            expiresAt: new Date(),
        };

        await repository.create(data);

        expect(mockDb.insert).toHaveBeenCalled();
        expect(mockDb.values).toHaveBeenCalledWith(data);
    });

    it('should find active by user id', async () => {
        const userId = '00000000-0000-0000-0000-000000000001';
        mockDb.limit.mockResolvedValue([{ id: 'test-id', userId }]);

        const result = await repository.findActiveByUserId(userId);

        expect(mockDb.select).toHaveBeenCalled();
        expect(mockDb.where).toHaveBeenCalled();
        expect(result).toEqual({ id: 'test-id', userId });
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
        const data: Partial<EmailChangeChallengeInsert> = { usedAt };

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
