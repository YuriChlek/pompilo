'use client';

import { useState } from 'react';
import {
    Alert,
    Box,
    Button,
    Chip,
    CircularProgress,
    Container,
    FormControl,
    InputLabel,
    MenuItem,
    Select,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    TextField,
    Typography,
} from '@mui/material';
import { GenericBotConfigForm } from '@/features/module-admin-bots/components/generic-bot-config-form';
import {
    useAdminBotConfigSchema,
    useAdminBotInstances,
    useAdminBotModules,
    useCreateAdminBotInstance,
    useEnableAdminBotInstance,
    usePauseAdminBotInstance,
    useRunAdminBotInstance,
    useValidateAdminBotConfig,
} from '@/features/module-admin-bots/hooks/use-admin-bots.hooks';
import type {
    AdminBotInstanceSummary,
    BotConfigValue,
    ManualBotRunResult,
} from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';

export const AdminBotConfigPage = () => {
    const [selectedModuleId, setSelectedModuleId] = useState<string | null>(null);
    const [instanceName, setInstanceName] = useState('');
    const [saveMessage, setSaveMessage] = useState<string | null>(null);
    const [lastRunByInstance, setLastRunByInstance] = useState<Record<string, ManualBotRunResult>>({});
    const { data: modules = [], isLoading: modulesLoading, error: modulesError } = useAdminBotModules();
    const { data: instances = [], isLoading: instancesLoading, error: instancesError } = useAdminBotInstances();
    const activeModuleId = selectedModuleId ?? modules[0]?.module_id ?? null;
    const {
        data: schemaResponse,
        isLoading: schemaLoading,
        error: schemaError,
    } = useAdminBotConfigSchema(activeModuleId);
    const validateConfig = useValidateAdminBotConfig();
    const createInstance = useCreateAdminBotInstance();
    const enableInstance = useEnableAdminBotInstance();
    const pauseInstance = usePauseAdminBotInstance();
    const runInstance = useRunAdminBotInstance();

    const selectedModule = modules.find(module => module.module_id === activeModuleId);
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
                {instancesError && <Alert severity="error">Failed to load bot instances.</Alert>}
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
                            value={activeModuleId ?? ''}
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
                    <Stack spacing={2}>
                        <Box>
                            <Typography variant="subtitle1">{selectedModule.display_name}</Typography>
                            <Typography variant="body2" color="text.secondary">
                                Version {selectedModule.version} · Schema{' '}
                                {selectedModule.config_schema_version ?? 'not available'}
                            </Typography>
                        </Box>
                        <TextField
                            size="small"
                            label="Instance name"
                            value={instanceName}
                            onChange={event => setInstanceName(event.target.value)}
                            sx={{ maxWidth: 420 }}
                        />
                    </Stack>
                )}

                {schemaLoading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', py: 6 }}>
                        <CircularProgress />
                    </Box>
                )}

                {!schemaLoading && activeModuleId && !schema && (
                    <Alert severity="warning">This module does not expose a config schema.</Alert>
                )}

                {activeModuleId && schemaResponse?.config_schema_version && schema && (
                    <GenericBotConfigForm
                        key={activeModuleId}
                        schema={schema}
                        isSaving={validateConfig.isPending || createInstance.isPending}
                        onValidate={async config => {
                            const result = await validateConfig.mutateAsync({
                                moduleId: activeModuleId,
                                configSchemaVersion: schemaResponse.config_schema_version ?? schema.schema_version,
                                config,
                            });
                            return result.valid
                                ? []
                                : result.errors.map(error => `${error.field_path}: ${error.message}`);
                        }}
                        onSubmit={async (config: Record<string, BotConfigValue>) => {
                            const result = await createInstance.mutateAsync({
                                moduleId: activeModuleId,
                                name: instanceName.trim() || `${selectedModule?.display_name ?? activeModuleId} instance`,
                                mode: 'signal_only',
                                symbols: resolveSymbols(config),
                                timeframes: resolveTimeframes(config, selectedModule?.required_timeframes ?? []),
                                configSchemaVersion: schemaResponse.config_schema_version ?? schema.schema_version,
                                config,
                            });
                            setSaveMessage(
                                result.accepted
                                    ? `Instance ${result.instance_id} created with status ${result.status}.`
                                    : `Instance create rejected: ${result.error_code ?? 'unknown error'}.`,
                            );
                        }}
                    />
                )}

                <Stack spacing={1.5}>
                    <Typography variant="h6">Bot instances</Typography>
                    {instancesLoading ? (
                        <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                            <CircularProgress />
                        </Box>
                    ) : (
                        <BotInstancesTable
                            instances={instances}
                            lastRunByInstance={lastRunByInstance}
                            isMutating={
                                enableInstance.isPending || pauseInstance.isPending || runInstance.isPending
                            }
                            onEnable={async instanceId => {
                                const result = await enableInstance.mutateAsync(instanceId);
                                setSaveMessage(
                                    result.accepted
                                        ? `Instance ${instanceId} enabled.`
                                        : `Enable rejected: ${result.error_code ?? 'unknown error'}.`,
                                );
                            }}
                            onPause={async instanceId => {
                                const result = await pauseInstance.mutateAsync(instanceId);
                                setSaveMessage(
                                    result.accepted
                                        ? `Instance ${instanceId} paused.`
                                        : `Pause rejected: ${result.error_code ?? 'unknown error'}.`,
                                );
                            }}
                            onRun={async instanceId => {
                                const result = await runInstance.mutateAsync(instanceId);
                                setLastRunByInstance(prev => ({ ...prev, [instanceId]: result }));
                                setSaveMessage(
                                    result.accepted
                                        ? `Manual run ${result.run_id} finished with ${result.status}.`
                                        : `Manual run rejected: ${result.error_code ?? 'unknown error'}.`,
                                );
                            }}
                        />
                    )}
                </Stack>
            </Stack>
        </Container>
    );
};

interface BotInstancesTableProps {
    instances: AdminBotInstanceSummary[];
    lastRunByInstance: Record<string, ManualBotRunResult>;
    isMutating: boolean;
    onEnable: (instanceId: string) => Promise<void>;
    onPause: (instanceId: string) => Promise<void>;
    onRun: (instanceId: string) => Promise<void>;
}

function BotInstancesTable({
    instances,
    lastRunByInstance,
    isMutating,
    onEnable,
    onPause,
    onRun,
}: BotInstancesTableProps) {
    if (instances.length === 0) {
        return <Alert severity="info">No bot instances configured.</Alert>;
    }

    return (
        <Box sx={{ overflowX: 'auto' }}>
            <Table size="small" aria-label="Bot instances">
                <TableHead>
                    <TableRow>
                        <TableCell>Name</TableCell>
                        <TableCell>Module</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Symbols</TableCell>
                        <TableCell>Timeframes</TableCell>
                        <TableCell>Last run</TableCell>
                        <TableCell align="right">Actions</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {instances.map(instance => {
                        const lastRun = lastRunByInstance[instance.instance_id];
                        return (
                            <TableRow key={instance.instance_id}>
                                <TableCell>{instance.name}</TableCell>
                                <TableCell>{instance.module_id}</TableCell>
                                <TableCell>
                                    <Chip label={instance.status} size="small" />
                                </TableCell>
                                <TableCell>{instance.symbols.join(', ')}</TableCell>
                                <TableCell>{instance.timeframes.join(', ')}</TableCell>
                                <TableCell>
                                    {lastRun
                                        ? `${lastRun.run_id ?? 'no run id'} · ${lastRun.status ?? lastRun.error_code ?? 'pending'}`
                                        : '-'}
                                </TableCell>
                                <TableCell align="right">
                                    <Stack direction="row" spacing={1} justifyContent="flex-end">
                                        <Button
                                            size="small"
                                            variant="outlined"
                                            disabled={isMutating || instance.status === 'ENABLED'}
                                            onClick={() => void onEnable(instance.instance_id)}
                                        >
                                            Enable
                                        </Button>
                                        <Button
                                            size="small"
                                            variant="outlined"
                                            disabled={isMutating || !['ENABLED', 'RUNNING'].includes(instance.status)}
                                            onClick={() => void onPause(instance.instance_id)}
                                        >
                                            Pause
                                        </Button>
                                        <Button
                                            size="small"
                                            variant="contained"
                                            disabled={isMutating || instance.status !== 'ENABLED'}
                                            onClick={() => void onRun(instance.instance_id)}
                                        >
                                            Run
                                        </Button>
                                    </Stack>
                                </TableCell>
                            </TableRow>
                        );
                    })}
                </TableBody>
            </Table>
        </Box>
    );
}

function resolveSymbols(config: Record<string, BotConfigValue>): string[] {
    const symbols = config.symbols;
    if (Array.isArray(symbols)) {
        return symbols.filter((value): value is string => typeof value === 'string' && Boolean(value.trim()));
    }
    return ['ETHUSDT'];
}

function resolveTimeframes(config: Record<string, BotConfigValue>, fallback: string[]): string[] {
    const timeframes = config.timeframes;
    if (Array.isArray(timeframes)) {
        return timeframes.filter((value): value is string => typeof value === 'string' && Boolean(value.trim()));
    }
    const primaryTimeframe = config.primary_timeframe;
    if (typeof primaryTimeframe === 'string' && primaryTimeframe.trim()) {
        return [primaryTimeframe];
    }
    return fallback.length > 0 ? fallback : ['1h'];
}
