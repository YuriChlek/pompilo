import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { adminMailApiService } from '../api-service/client';
import { SendTestEmailDto, UpdateMailSettingsDto } from '../interfaces/admin-mail.interfaces';

export const adminMailQueryKeys = {
    settings: ['admin', 'mail', 'settings'] as const,
    setupState: ['admin', 'mail', 'setup-state'] as const,
    templates: ['admin', 'mail', 'templates'] as const,
    templatePreview: (templateId: string) =>
        ['admin', 'mail', 'templates', templateId, 'preview'] as const,
};

export const useMailSettings = () => {
    return useQuery({
        queryKey: adminMailQueryKeys.settings,
        queryFn: () => adminMailApiService.getSettings(),
    });
};

export const useMailSetupState = () => {
    return useQuery({
        queryKey: adminMailQueryKeys.setupState,
        queryFn: () => adminMailApiService.getSetupState(),
    });
};

export const useUpdateMailSettings = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (dto: UpdateMailSettingsDto) => adminMailApiService.updateSettings(dto),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: adminMailQueryKeys.settings });
            queryClient.invalidateQueries({ queryKey: adminMailQueryKeys.setupState });
        },
    });
};

export const useSendTestMail = () => {
    return useMutation({
        mutationFn: (dto: SendTestEmailDto) => adminMailApiService.sendTestEmail(dto),
    });
};

export const useMailTemplates = () => {
    return useQuery({
        queryKey: adminMailQueryKeys.templates,
        queryFn: () => adminMailApiService.getTemplates(),
        staleTime: 300000,
    });
};

export const useMailTemplatePreview = (templateId: string | null) => {
    return useQuery({
        queryKey: adminMailQueryKeys.templatePreview(templateId ?? 'none'),
        queryFn: () => adminMailApiService.getTemplatePreview(templateId as string),
        enabled: Boolean(templateId),
        staleTime: 300000,
    });
};
