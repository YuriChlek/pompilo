import { useMutation, useQuery } from '@tanstack/react-query';
import { adminBotsApiService } from '@/features/module-admin-bots/api-service/client';
import type { ValidateBotConfigDto } from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';

export const adminBotsQueryKeys = {
    modules: ['admin', 'bot-modules'] as const,
    configSchema: (moduleId: string | null) =>
        ['admin', 'bot-modules', moduleId ?? 'none', 'config-schema'] as const,
};

export const useAdminBotModules = () => {
    return useQuery({
        queryKey: adminBotsQueryKeys.modules,
        queryFn: () => adminBotsApiService.getModules(),
        staleTime: 300000,
    });
};

export const useAdminBotConfigSchema = (moduleId: string | null) => {
    return useQuery({
        queryKey: adminBotsQueryKeys.configSchema(moduleId),
        queryFn: () => adminBotsApiService.getConfigSchema(moduleId as string),
        enabled: Boolean(moduleId),
        staleTime: 300000,
    });
};

export const useValidateAdminBotConfig = () => {
    return useMutation({
        mutationFn: (dto: ValidateBotConfigDto) => adminBotsApiService.validateConfig(dto),
    });
};
