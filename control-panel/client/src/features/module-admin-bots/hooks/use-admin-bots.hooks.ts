import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { adminBotsApiService } from '@/features/module-admin-bots/api-service/client';
import type {
    CreateBotInstanceDto,
    ValidateBotConfigDto,
} from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';

export const adminBotsQueryKeys = {
    modules: ['admin', 'bot-modules'] as const,
    instances: ['admin', 'bot-instances'] as const,
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

export const useAdminBotInstances = () => {
    return useQuery({
        queryKey: adminBotsQueryKeys.instances,
        queryFn: () => adminBotsApiService.getInstances(),
        staleTime: 30000,
    });
};

export const useCreateAdminBotInstance = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (dto: CreateBotInstanceDto) => adminBotsApiService.createInstance(dto),
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: adminBotsQueryKeys.instances });
        },
    });
};

export const useEnableAdminBotInstance = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (instanceId: string) => adminBotsApiService.enableInstance(instanceId),
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: adminBotsQueryKeys.instances });
        },
    });
};

export const usePauseAdminBotInstance = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (instanceId: string) => adminBotsApiService.pauseInstance(instanceId),
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: adminBotsQueryKeys.instances });
        },
    });
};

export const useRunAdminBotInstance = () => {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: (instanceId: string) => adminBotsApiService.runInstance(instanceId),
        onSuccess: async () => {
            await queryClient.invalidateQueries({ queryKey: adminBotsQueryKeys.instances });
        },
    });
};
