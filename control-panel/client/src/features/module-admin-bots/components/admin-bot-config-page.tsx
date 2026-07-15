'use client';

import { useEffect, useState } from 'react';
import {
    Alert,
    Box,
    CircularProgress,
    Container,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    Typography,
} from '@mui/material';
import { GenericBotConfigForm } from '@/features/module-admin-bots/components/generic-bot-config-form';
import {
    useAdminBotConfigSchema,
    useAdminBotModules,
    useValidateAdminBotConfig,
} from '@/features/module-admin-bots/hooks/use-admin-bots.hooks';
import type { BotConfigValue } from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';

export const AdminBotConfigPage = () => {
    const [selectedModuleId, setSelectedModuleId] = useState<string | null>(null);
    const [saveMessage, setSaveMessage] = useState<string | null>(null);
    const { data: modules = [], isLoading: modulesLoading, error: modulesError } = useAdminBotModules();
    const {
        data: schemaResponse,
        isLoading: schemaLoading,
        error: schemaError,
    } = useAdminBotConfigSchema(selectedModuleId);
    const validateConfig = useValidateAdminBotConfig();

    useEffect(() => {
        if (!selectedModuleId && modules.length > 0) {
            setSelectedModuleId(modules[0].module_id);
        }
    }, [modules, selectedModuleId]);

    const selectedModule = modules.find(module => module.module_id === selectedModuleId);
    const schema = schemaResponse?.config_schema ?? null;

    return (
        <Container maxWidth="lg" sx={{ py: 4 }}>
            <Stack spacing={3}>
                <Box>
                    <Typography variant="h4" component="h1">
                        Bot Configuration
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75 }}>
                        Schema-driven configuration for registered trading bot modules.
                    </Typography>
                </Box>

                {modulesError && <Alert severity="error">Failed to load bot modules.</Alert>}
                {schemaError && <Alert severity="error">Failed to load module config schema.</Alert>}
                {saveMessage && <Alert severity="success">{saveMessage}</Alert>}

                {modulesLoading ? (
                    <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                        <CircularProgress />
                    </Box>
                ) : (
                    <FormControl size="small" sx={{ maxWidth: 420 }}>
                        <InputLabel>Module</InputLabel>
                        <Select
                            label="Module"
                            value={selectedModuleId ?? ''}
                            onChange={event => {
                                setSaveMessage(null);
                                setSelectedModuleId(event.target.value);
                            }}
                        >
                            {modules.map(module => (
                                <MenuItem key={module.module_id} value={module.module_id}>
                                    {module.display_name}
                                </MenuItem>
                            ))}
                        </Select>
                    </FormControl>
                )}

                {selectedModule && (
                    <Box>
                        <Typography variant="subtitle1">{selectedModule.display_name}</Typography>
                        <Typography variant="body2" color="text.secondary">
                            Version {selectedModule.version} · Schema{' '}
                            {selectedModule.config_schema_version ?? 'not available'}
                        </Typography>
                    </Box>
                )}

                {schemaLoading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
                        <CircularProgress />
                    </Box>
                )}

                {!schemaLoading && selectedModuleId && !schema && (
                    <Alert severity="warning">This module does not expose a config schema.</Alert>
                )}

                {selectedModuleId && schemaResponse?.config_schema_version && schema && (
                    <GenericBotConfigForm
                        key={selectedModuleId}
                        schema={schema}
                        isSaving={validateConfig.isPending}
                        onValidate={async config => {
                            const result = await validateConfig.mutateAsync({
                                moduleId: selectedModuleId,
                                configSchemaVersion: schemaResponse.config_schema_version ?? schema.schema_version,
                                config,
                            });
                            return result.valid ? [] : result.errors;
                        }}
                        onSubmit={(config: Record<string, BotConfigValue>) => {
                            void config;
                            setSaveMessage('Configuration passed backend validation.');
                        }}
                    />
                )}
            </Stack>
        </Container>
    );
};
