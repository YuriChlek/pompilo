'use client';

import { FormEvent, useMemo, useState } from 'react';
import {
    Alert,
    Box,
    Button,
    CircularProgress,
    FormControl,
    FormControlLabel,
    FormHelperText,
    Grid,
    InputLabel,
    MenuItem,
    Paper,
    Select,
    Stack,
    Switch,
    TextField,
    Typography,
} from '@mui/material';
import type {
    BotConfigField,
    BotConfigSchema,
    BotConfigValue,
} from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';
import {
    buildInitialConfig,
    getConfigValue,
    parseListValue,
    setConfigValue,
    type ConfigPath,
} from '@/features/module-admin-bots/lib/config-schema-form';

interface GenericBotConfigFormProps {
    schema: BotConfigSchema;
    initialValue?: Record<string, BotConfigValue>;
    isSaving?: boolean;
    onValidate?: (config: Record<string, BotConfigValue>) => Promise<string[]> | string[];
    onSubmit: (config: Record<string, BotConfigValue>) => void;
}

export const GenericBotConfigForm = ({
    schema,
    initialValue,
    isSaving = false,
    onValidate,
    onSubmit,
}: GenericBotConfigFormProps) => {
    const initialConfig = useMemo(
        () => buildInitialConfig(schema, initialValue),
        [schema, initialValue],
    );
    const [config, setConfig] = useState<Record<string, BotConfigValue>>(initialConfig);
    const [validationErrors, setValidationErrors] = useState<string[]>([]);
    const [isValidating, setIsValidating] = useState(false);

    const updateValue = (path: ConfigPath, value: BotConfigValue) => {
        setConfig(prev => setConfigValue(prev, path, value));
    };

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        setValidationErrors([]);

        if (onValidate) {
            setIsValidating(true);
            try {
                const errors = await onValidate(config);
                if (errors.length > 0) {
                    setValidationErrors(errors);
                    return;
                }
            } finally {
                setIsValidating(false);
            }
        }

        onSubmit(config);
    };

    const disabled = isSaving || isValidating;

    return (
        <Box component="form" onSubmit={handleSubmit}>
            <Stack spacing={3}>
                {validationErrors.length > 0 && (
                    <Alert severity="error">
                        <Stack spacing={0.5}>
                            {validationErrors.map(error => (
                                <span key={error}>{error}</span>
                            ))}
                        </Stack>
                    </Alert>
                )}

                {schema.sections.map(section => (
                    <Box key={section.key}>
                        <Typography variant="h6">{section.label}</Typography>
                        {section.description && (
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                                {section.description}
                            </Typography>
                        )}
                        <Grid container spacing={2.5} sx={{ mt: 0.5 }}>
                            {section.fields.map(field => (
                                <Grid item xs={12} md={field.type === 'boolean' ? 12 : 6} key={field.key}>
                                    <ConfigFieldControl
                                        field={field}
                                        path={[field.key]}
                                        value={getConfigValue(config, [field.key])}
                                        disabled={disabled}
                                        onChange={updateValue}
                                    />
                                </Grid>
                            ))}
                        </Grid>
                    </Box>
                ))}

                <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
                    <Button
                        type="submit"
                        variant="contained"
                        size="small"
                        disabled={disabled}
                        startIcon={disabled ? <CircularProgress color="inherit" size={18} /> : null}
                    >
                        {isValidating ? 'Validating...' : isSaving ? 'Saving...' : 'Validate and save'}
                    </Button>
                </Box>
            </Stack>
        </Box>
    );
};

interface ConfigFieldControlProps {
    field: BotConfigField;
    path: ConfigPath;
    value: BotConfigValue;
    disabled: boolean;
    onChange: (path: ConfigPath, value: BotConfigValue) => void;
}

function ConfigFieldControl({
    field,
    path,
    value,
    disabled,
    onChange,
}: ConfigFieldControlProps) {
    const helperText = field.description || (field.required ? 'Required' : undefined);

    if (field.type === 'boolean') {
        return (
            <FormControlLabel
                control={
                    <Switch
                        checked={Boolean(value)}
                        onChange={event => onChange(path, event.target.checked)}
                        disabled={disabled}
                    />
                }
                label={field.label}
            />
        );
    }

    if (field.type === 'enum') {
        const labelId = `${path.join('-')}-label`;
        return (
            <FormControl fullWidth size="small" disabled={disabled}>
                <InputLabel id={labelId}>{field.label}</InputLabel>
                <Select
                    labelId={labelId}
                    label={field.label}
                    value={typeof value === 'string' ? value : ''}
                    onChange={event => onChange(path, event.target.value)}
                >
                    {(field.allowed ?? []).map(option => (
                        <MenuItem key={option} value={option}>
                            {option}
                        </MenuItem>
                    ))}
                </Select>
                {helperText && <FormHelperText>{helperText}</FormHelperText>}
            </FormControl>
        );
    }

    if (field.type === 'object') {
        return (
            <Paper variant="outlined" sx={{ p: 2, borderRadius: 1 }}>
                <Typography variant="subtitle2">{field.label}</Typography>
                {field.description && (
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                        {field.description}
                    </Typography>
                )}
                <Stack spacing={2} sx={{ mt: 2 }}>
                    {(field.fields ?? []).map(child => (
                        <ConfigFieldControl
                            key={child.key}
                            field={child}
                            path={[...path, child.key]}
                            value={getNestedValue(value, child.key)}
                            disabled={disabled}
                            onChange={onChange}
                        />
                    ))}
                </Stack>
            </Paper>
        );
    }

    if (field.type === 'array') {
        const arrayValue = Array.isArray(value) ? value : [];
        return (
            <TextField
                fullWidth
                size="small"
                label={field.label}
                value={arrayValue.join(', ')}
                helperText={helperText || 'Comma separated values'}
                disabled={disabled}
                onChange={event => onChange(path, parseListValue(event.target.value))}
            />
        );
    }

    if (field.type === 'symbol_list' || field.type === 'timeframe_list') {
        const listValue = Array.isArray(value) ? value : [];
        return (
            <TextField
                fullWidth
                size="small"
                label={field.label}
                value={listValue.join(', ')}
                helperText={helperText || 'Comma separated values'}
                disabled={disabled}
                onChange={event => onChange(path, parseListValue(event.target.value))}
            />
        );
    }

    const type = field.type === 'integer' ? 'number' : field.type === 'secret_ref' ? 'password' : 'text';

    return (
        <TextField
            fullWidth
            size="small"
            type={type}
            label={field.label}
            value={formatScalarValue(value)}
            helperText={helperText}
            disabled={disabled}
            inputProps={{
                min: field.min,
                max: field.max,
                step: field.step,
            }}
            onChange={event => {
                const nextValue =
                    field.type === 'integer'
                        ? parseIntegerValue(event.target.value)
                        : event.target.value;
                onChange(path, nextValue);
            }}
        />
    );
}

function getNestedValue(value: BotConfigValue, key: string): BotConfigValue {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) {
        return null;
    }
    return value[key] ?? null;
}

function formatScalarValue(value: BotConfigValue): string | number {
    if (typeof value === 'string' || typeof value === 'number') {
        return value;
    }
    return '';
}

function parseIntegerValue(value: string): number | string {
    if (!value.trim()) {
        return '';
    }
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? '' : parsed;
}
