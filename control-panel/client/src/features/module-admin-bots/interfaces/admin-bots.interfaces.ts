export type BotConfigFieldType =
    | 'string'
    | 'integer'
    | 'decimal'
    | 'boolean'
    | 'enum'
    | 'symbol'
    | 'symbol_list'
    | 'timeframe'
    | 'timeframe_list'
    | 'secret_ref'
    | 'object'
    | 'array';

export type BotConfigValue =
    | string
    | number
    | boolean
    | null
    | BotConfigValue[]
    | { [key: string]: BotConfigValue };

export interface BotConfigField {
    key: string;
    type: BotConfigFieldType;
    label: string;
    description?: string;
    required?: boolean;
    default?: BotConfigValue;
    allowed?: string[];
    min?: string | number;
    max?: string | number;
    step?: string | number;
    fields?: BotConfigField[];
    items?: BotConfigField;
}

export interface BotConfigSection {
    key: string;
    label: string;
    description?: string;
    fields: BotConfigField[];
}

export interface BotConfigSchema {
    schema_version: number;
    sections: BotConfigSection[];
}

export interface AdminBotModuleSummary {
    module_id: string;
    display_name: string;
    version: string;
    status: string;
    supported_modes: string[];
    required_timeframes: string[];
    required_market_data: string[];
    supports_multi_symbol: boolean;
    config_schema_version: number | null;
    config_schema_available: boolean;
}

export interface AdminBotConfigSchemaResponse {
    module_id: string;
    config_schema_version: number | null;
    config_schema: BotConfigSchema | null;
}

export interface ValidateBotConfigDto {
    moduleId: string;
    configSchemaVersion: number;
    config: Record<string, BotConfigValue>;
}

export interface ValidateBotConfigResult {
    valid: boolean;
    errors: BotConfigValidationError[];
}

export interface BotConfigValidationError {
    field_path: string;
    code: string;
    message: string;
}

export interface AdminBotInstanceSummary {
    instance_id: string;
    module_id: string;
    tenant_id: string | null;
    name: string;
    mode: string;
    status: string;
    symbols: string[];
    timeframes: string[];
    config_schema_version: number;
    config: Record<string, BotConfigValue>;
}

export interface CreateBotInstanceDto {
    instanceId?: string;
    moduleId: string;
    name?: string;
    mode: string;
    symbols: string[];
    timeframes: string[];
    configSchemaVersion: number;
    config: Record<string, BotConfigValue>;
}

export interface BotInstanceActionResult {
    accepted: boolean;
    instance_id: string;
    status: string | null;
    error_code: string | null;
}

export interface ManualBotRunResult {
    accepted: boolean;
    instance_id: string;
    run_id: string | null;
    status: string | null;
    error_code: string | null;
    duplicate: boolean;
}
