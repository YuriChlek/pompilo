import { Test, TestingModule } from '@nestjs/testing';
import { MailOutboxCleanupService } from '@/module-mail/services/mail-outbox-cleanup.service';
import { MailOutboxRepository } from '@/module-mail/repository/mail-outbox.repository';

describe('MailOutboxCleanupService', () => {
    let service: MailOutboxCleanupService;
    let mockOutboxRepository: {
        deleteOldRecordsByStatusInBatches: jest.Mock;
    };

    beforeEach(async () => {
        mockOutboxRepository = {
            deleteOldRecordsByStatusInBatches: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailOutboxCleanupService,
                {
                    provide: MailOutboxRepository,
                    useValue: mockOutboxRepository,
                },
            ],
        }).compile();

        service = module.get<MailOutboxCleanupService>(MailOutboxCleanupService);
    });

    it('should successfully cleanup sent and failed outbox records', async () => {
        mockOutboxRepository.deleteOldRecordsByStatusInBatches
            .mockResolvedValueOnce(5) // sent
            .mockResolvedValueOnce(2); // failed

        await service.cleanup();

        expect(mockOutboxRepository.deleteOldRecordsByStatusInBatches).toHaveBeenCalledTimes(2);

        const calls = mockOutboxRepository.deleteOldRecordsByStatusInBatches.mock
            .calls as unknown as ['sent' | 'failed', Date][];

        // Check first call (sent)
        const firstCall = calls[0];
        expect(firstCall[0]).toBe('sent');
        // date should be around 24 hours ago
        const expectedSentDate = new Date();
        expectedSentDate.setHours(expectedSentDate.getHours() - 24);
        expect(Math.abs(firstCall[1].getTime() - expectedSentDate.getTime())).toBeLessThan(5000); // within 5s

        // Check second call (failed)
        const secondCall = calls[1];
        expect(secondCall[0]).toBe('failed');
        // date should be around 7 days ago
        const expectedFailedDate = new Date();
        expectedFailedDate.setDate(expectedFailedDate.getDate() - 7);
        expect(Math.abs(secondCall[1].getTime() - expectedFailedDate.getTime())).toBeLessThan(5000); // within 5s
    });

    it('should handle repository errors gracefully without throwing', async () => {
        mockOutboxRepository.deleteOldRecordsByStatusInBatches.mockRejectedValue(
            new Error('DB error'),
        );

        await expect(service.cleanup()).resolves.not.toThrow();
    });
});
