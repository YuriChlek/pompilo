import { apiClient } from '@/lib/http-client/http-client';
import {
    MailSettings,
    SendTestEmailDto,
    SendTestEmailResponse,
    UpdateMailSettingsDto,
} from '../../interfaces/admin-mail.interfaces';
import { MailSetupState } from '../../types/admin-mail.types';
import {
    MailTemplatePreview,
    MailTemplateSummary,
} from '../../interfaces/mail-template-preview.interfaces';

export const adminMailApiService = {
    async getSettings(): Promise<MailSettings | null> {
        const response = await apiClient.get<MailSettings>('/admin/mail/settings');
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as MailSettings;
    },

    async getSetupState(): Promise<{ state: MailSetupState }> {
        const response = await apiClient.get<{ state: MailSetupState }>('/admin/mail/setup-state');
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as { state: MailSetupState };
    },

    async updateSettings(dto: UpdateMailSettingsDto): Promise<MailSettings> {
        const response = await apiClient.patch<MailSettings, UpdateMailSettingsDto>(
            '/admin/mail/settings',
            dto,
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as MailSettings;
    },

    async sendTestEmail(dto: SendTestEmailDto): Promise<SendTestEmailResponse> {
        const response = await apiClient.post<SendTestEmailResponse, SendTestEmailDto>(
            '/admin/mail/send-test-email',
            dto,
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as SendTestEmailResponse;
    },

    async getTemplates(): Promise<MailTemplateSummary[]> {
        const response = await apiClient.get<MailTemplateSummary[]>('/admin/mail/templates');
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as MailTemplateSummary[];
    },

    async getTemplatePreview(templateId: string): Promise<MailTemplatePreview> {
        const response = await apiClient.get<MailTemplatePreview>(
            `/admin/mail/templates/${templateId}/preview`,
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as MailTemplatePreview;
    },
};
