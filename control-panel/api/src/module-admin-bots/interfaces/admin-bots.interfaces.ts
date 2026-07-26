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

export interface BotConfigSchema {
    schema_version: number;
    sections: unknown[];
}

export interface AdminBotConfigSchemaResponse {
    module_id: string;
    config_schema_version: number | null;
    config_schema: BotConfigSchema | null;
}

export interface BotPlatformModuleListResponse {
    modules: AdminBotModuleSummary[];
}

export interface ValidateBotConfigDto {
    moduleId: string;
    configSchemaVersion: number;
    config: Record<string, unknown>;
}

export interface BotConfigValidationError {
    field_path: string;
    code: string;
    message: string;
}

export interface ValidateBotConfigResult {
    valid: boolean;
    errors: BotConfigValidationError[];
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
    config: Record<string, unknown>;
}

export interface BotPlatformInstanceListResponse {
    instances: AdminBotInstanceSummary[];
}

export interface CreateBotInstanceDto {
    instanceId?: string;
    moduleId: string;
    name?: string;
    mode: string;
    symbols: string[];
    timeframes: string[];
    configSchemaVersion: number;
    config: Record<string, unknown>;
}

export interface RunBotInstanceDto {
    idempotencyKey?: string;
    correlationId?: string;
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
