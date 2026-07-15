import { describe, it, expect, vi, beforeEach } from 'vitest';
import { adminMailApiService } from '@/features/module-admin-mail/api-service/client';
import { apiClient } from '@/lib/http-client/http-client';

vi.mock('@/lib/http-client/http-client', () => ({
    apiClient: {
        post: vi.fn(),
        get: vi.fn(),
    },
}));

describe('adminMailApiService', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('sendTestEmail calls the send-test-email endpoint with recipient and template id', async () => {
        vi.mocked(apiClient.post).mockResolvedValue({
            success: true,
            statusCode: 200,
            timestamp: '2026-01-01T00:00:00.000Z',
            data: { outboxId: 'outbox-1' },
        });

        const result = await adminMailApiService.sendTestEmail({
            to: 'recipient@example.com',
            templateId: 'security-alert',
        });

        expect(apiClient.post).toHaveBeenCalledWith('/admin/mail/send-test-email', {
            to: 'recipient@example.com',
            templateId: 'security-alert',
        });
        expect(result).toEqual({ outboxId: 'outbox-1' });
    });

    it('getTemplates calls the templates endpoint and returns templates', async () => {
        const mockTemplates = [{ id: 't1', name: 'T1', description: 'Desc1', subject: 'Subj1' }];
        vi.mocked(apiClient.get).mockResolvedValue({
            success: true,
            statusCode: 200,
            timestamp: '2026-01-01T00:00:00.000Z',
            data: mockTemplates,
        });

        const result = await adminMailApiService.getTemplates();

        expect(apiClient.get).toHaveBeenCalledWith('/admin/mail/templates');
        expect(result).toEqual(mockTemplates);
    });

    it('getTemplatePreview calls the preview endpoint and returns rendering', async () => {
        const mockPreview = {
            id: 't1',
            name: 'T1',
            description: 'Desc1',
            subject: 'Subj1',
            html: 'HTML',
            text: 'text',
        };
        vi.mocked(apiClient.get).mockResolvedValue({
            success: true,
            statusCode: 200,
            timestamp: '2026-01-01T00:00:00.000Z',
            data: mockPreview,
        });

        const result = await adminMailApiService.getTemplatePreview('t1');

        expect(apiClient.get).toHaveBeenCalledWith('/admin/mail/templates/t1/preview');
        expect(result).toEqual(mockPreview);
    });
});
