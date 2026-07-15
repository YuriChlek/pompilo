import { apiClient } from '@/lib/http-client/http-client';
import type {
    AdminBotConfigSchemaResponse,
    AdminBotModuleSummary,
    ValidateBotConfigDto,
    ValidateBotConfigResult,
} from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';

export const adminBotsApiService = {
    async getModules(): Promise<AdminBotModuleSummary[]> {
        const response = await apiClient.get<AdminBotModuleSummary[]>('/admin/bot-modules');
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as AdminBotModuleSummary[];
    },

    async getConfigSchema(moduleId: string): Promise<AdminBotConfigSchemaResponse> {
        const response = await apiClient.get<AdminBotConfigSchemaResponse>(
            `/admin/bot-modules/${moduleId}/config-schema`,
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as AdminBotConfigSchemaResponse;
    },

    async validateConfig(dto: ValidateBotConfigDto): Promise<ValidateBotConfigResult> {
        const response = await apiClient.post<ValidateBotConfigResult, ValidateBotConfigDto>(
            '/admin/bot-instances/validate-config',
            dto,
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as ValidateBotConfigResult;
    },
};
