import { apiClient } from '@/lib/http-client/http-client';
import type {
    AdminBotConfigSchemaResponse,
    AdminBotInstanceSummary,
    AdminBotModuleSummary,
    BotInstanceActionResult,
    CreateBotInstanceDto,
    ManualBotRunResult,
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
            `/admin/bot-modules/${encodeURIComponent(moduleId)}/config-schema`,
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

    async getInstances(): Promise<AdminBotInstanceSummary[]> {
        const response = await apiClient.get<AdminBotInstanceSummary[]>('/admin/bot-instances');
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as AdminBotInstanceSummary[];
    },

    async createInstance(dto: CreateBotInstanceDto): Promise<BotInstanceActionResult> {
        const response = await apiClient.post<BotInstanceActionResult, CreateBotInstanceDto>(
            '/admin/bot-instances',
            dto,
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as BotInstanceActionResult;
    },

    async enableInstance(instanceId: string): Promise<BotInstanceActionResult> {
        const response = await apiClient.post<BotInstanceActionResult, Record<string, never>>(
            `/admin/bot-instances/${encodeURIComponent(instanceId)}/enable`,
            {},
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as BotInstanceActionResult;
    },

    async pauseInstance(instanceId: string): Promise<BotInstanceActionResult> {
        const response = await apiClient.post<BotInstanceActionResult, Record<string, never>>(
            `/admin/bot-instances/${encodeURIComponent(instanceId)}/pause`,
            {},
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as BotInstanceActionResult;
    },

    async runInstance(instanceId: string): Promise<ManualBotRunResult> {
        const response = await apiClient.post<ManualBotRunResult, Record<string, never>>(
            `/admin/bot-instances/${encodeURIComponent(instanceId)}/run`,
            {},
        );
        if (!response.success) {
            throw new Error(response.message);
        }
        return response.data as unknown as ManualBotRunResult;
    },
};
