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
    errors: string[];
}
