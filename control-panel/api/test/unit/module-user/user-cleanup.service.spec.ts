import { Logger } from '@nestjs/common';
import { UserCleanupService } from '@/module-user/services/user-cleanup.service';
import { UserRepository } from '@/module-user/repository/user.repository';

describe('UserCleanupService', () => {
    let service: UserCleanupService;
    let loggerErrorSpy: jest.SpiedFunction<Logger['error']>;
    let loggerLogSpy: jest.SpiedFunction<Logger['log']>;
    let mockUserRepository: {
        deleteExpiredUsers: jest.Mock;
    };

    beforeEach(() => {
        loggerErrorSpy = jest.spyOn(Logger.prototype, 'error').mockImplementation(() => {});
        loggerLogSpy = jest.spyOn(Logger.prototype, 'log').mockImplementation(() => {});

        mockUserRepository = {
            deleteExpiredUsers: jest.fn(),
        };

        service = new UserCleanupService(mockUserRepository as unknown as UserRepository);
    });

    afterEach(() => {
        loggerErrorSpy.mockRestore();
        loggerLogSpy.mockRestore();
    });

    it('successfully purges expired users', async () => {
        mockUserRepository.deleteExpiredUsers.mockResolvedValue(5);

        await service.handleCron();

        expect(mockUserRepository.deleteExpiredUsers).toHaveBeenCalled();
    });

    it('handles database errors gracefully during cleanup', async () => {
        const error = new Error('Database connection failed');
        mockUserRepository.deleteExpiredUsers.mockRejectedValue(error);

        // Should not throw, should log instead
        await expect(service.handleCron()).resolves.not.toThrow();
        expect(mockUserRepository.deleteExpiredUsers).toHaveBeenCalled();
    });
});
