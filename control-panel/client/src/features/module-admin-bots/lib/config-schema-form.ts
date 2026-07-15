import type {
    BotConfigField,
    BotConfigSchema,
    BotConfigValue,
} from '@/features/module-admin-bots/interfaces/admin-bots.interfaces';

export type ConfigPath = string[];

export function buildInitialConfig(
    schema: BotConfigSchema,
    current: Record<string, BotConfigValue> = {},
): Record<string, BotConfigValue> {
    const values: Record<string, BotConfigValue> = { ...current };

    for (const section of schema.sections) {
        for (const field of section.fields) {
            if (values[field.key] === undefined) {
                values[field.key] = defaultValueForField(field);
            }
        }
    }

    return values;
}

export function defaultValueForField(field: BotConfigField): BotConfigValue {
    if (field.default !== undefined) {
        return cloneValue(field.default);
    }

    switch (field.type) {
        case 'integer':
            return 0;
        case 'boolean':
            return false;
        case 'symbol_list':
        case 'timeframe_list':
        case 'array':
            return [];
        case 'object':
            return buildObjectDefault(field.fields ?? []);
        case 'decimal':
        case 'enum':
        case 'secret_ref':
        case 'string':
        case 'symbol':
        case 'timeframe':
            return '';
    }
}

export function getConfigValue(
    values: Record<string, BotConfigValue>,
    path: ConfigPath,
): BotConfigValue {
    let current: BotConfigValue | Record<string, BotConfigValue> = values;

    for (const segment of path) {
        if (!isConfigObject(current)) {
            return null;
        }
        current = current[segment] ?? null;
    }

    return current;
}

export function setConfigValue(
    values: Record<string, BotConfigValue>,
    path: ConfigPath,
    value: BotConfigValue,
): Record<string, BotConfigValue> {
    if (path.length === 0) {
        return values;
    }

    const [head, ...tail] = path;
    if (tail.length === 0) {
        return { ...values, [head]: value };
    }

    const current = values[head];
    const nested = isConfigObject(current) ? current : {};

    return {
        ...values,
        [head]: setNestedValue(nested, tail, value),
    };
}

export function parseListValue(value: string): string[] {
    return value
        .split(',')
        .map(item => item.trim())
        .filter(Boolean);
}

function setNestedValue(
    values: Record<string, BotConfigValue>,
    path: ConfigPath,
    value: BotConfigValue,
): Record<string, BotConfigValue> {
    const [head, ...tail] = path;

    if (tail.length === 0) {
        return { ...values, [head]: value };
    }

    const current = values[head];
    const nested = isConfigObject(current) ? current : {};

    return {
        ...values,
        [head]: setNestedValue(nested, tail, value),
    };
}

function buildObjectDefault(fields: BotConfigField[]): Record<string, BotConfigValue> {
    return fields.reduce<Record<string, BotConfigValue>>((result, field) => {
        result[field.key] = defaultValueForField(field);
        return result;
    }, {});
}

function cloneValue(value: BotConfigValue): BotConfigValue {
    if (Array.isArray(value)) {
        return value.map(item => cloneValue(item));
    }
    if (isConfigObject(value)) {
        return Object.fromEntries(
            Object.entries(value).map(([key, item]) => [key, cloneValue(item)]),
        );
    }
    return value;
}

function isConfigObject(value: unknown): value is Record<string, BotConfigValue> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
