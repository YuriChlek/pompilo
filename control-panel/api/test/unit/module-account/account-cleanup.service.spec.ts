import { Test, TestingModule } from '@nestjs/testing';
import { AccountCleanupService } from '@/module-account/services/account-cleanup.service';
import { PasswordResetChallengeRepository } from '@/module-account/repository/password-reset-challenge.repository';
import { EmailChangeChallengeRepository } from '@/module-account/repository/email-change-challenge.repository';

describe('AccountCleanupService', () => {
    let service: AccountCleanupService;
    let passwordResetRepository: { cleanup: jest.Mock };
    let emailChangeRepository: { cleanup: jest.Mock };

    beforeEach(async () => {
        passwordResetRepository = { cleanup: jest.fn() };
        emailChangeRepository = { cleanup: jest.fn() };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                AccountCleanupService,
                { provide: PasswordResetChallengeRepository, useValue: passwordResetRepository },
                { provide: EmailChangeChallengeRepository, useValue: emailChangeRepository },
            ],
        }).compile();

        service = module.get<AccountCleanupService>(AccountCleanupService);
    });

    it('should call cleanup on both repositories with correct retention period', async () => {
        passwordResetRepository.cleanup.mockResolvedValue(5);
        emailChangeRepository.cleanup.mockResolvedValue(3);

        await service.cleanupChallenges();

        expect(passwordResetRepository.cleanup).toHaveBeenCalledWith(expect.any(Date));
        expect(emailChangeRepository.cleanup).toHaveBeenCalledWith(expect.any(Date));

        const calls = passwordResetRepository.cleanup.mock.calls as [Date][];
        const callDate = calls[0][0];
        const now = new Date();
        const diffHours = (now.getTime() - callDate.getTime()) / (1000 * 60 * 60);

        // Should be approximately 24 hours ago
        expect(diffHours).toBeGreaterThan(23.9);
        expect(diffHours).toBeLessThan(24.1);
    });

    it('should handle repository errors gracefully', async () => {
        passwordResetRepository.cleanup.mockRejectedValue(new Error('DB error'));

        await expect(service.cleanupChallenges()).resolves.not.toThrow();
        expect(passwordResetRepository.cleanup).toHaveBeenCalled();
    });
});
