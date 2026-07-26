import { Injectable, NotFoundException, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
    AdminBotConfigSchemaResponse,
    AdminBotInstanceSummary,
    AdminBotModuleSummary,
    BotInstanceActionResult,
    BotPlatformInstanceListResponse,
    BotPlatformModuleListResponse,
    CreateBotInstanceDto,
    ManualBotRunResult,
    RunBotInstanceDto,
    ValidateBotConfigDto,
    ValidateBotConfigResult,
} from '@/module-admin-bots/interfaces/admin-bots.interfaces';

const DEFAULT_BOT_PLATFORM_BASE_URL = 'http://bot_platform:8092';

@Injectable()
export class AdminBotsService {
    constructor(private readonly configService: ConfigService) {}

    async listModules(): Promise<AdminBotModuleSummary[]> {
        const payload = await this.getJson<BotPlatformModuleListResponse>('/admin/bot-modules');

        if (!Array.isArray(payload.modules)) {
            throw new ServiceUnavailableException('bot_platform_modules_payload_invalid');
        }

        return payload.modules;
    }

    async getConfigSchema(moduleId: string): Promise<AdminBotConfigSchemaResponse> {
        const payload = await this.getJson<AdminBotConfigSchemaResponse>(
            `/admin/bot-modules/${encodeURIComponent(moduleId)}/config-schema`,
        );

        if (payload.module_id !== moduleId) {
            throw new ServiceUnavailableException('bot_platform_config_schema_payload_invalid');
        }

        return payload;
    }

    async validateConfig(dto: ValidateBotConfigDto): Promise<ValidateBotConfigResult> {
        const payload = await this.postJson<ValidateBotConfigResult>(
            '/admin/bot-instances/validate-config',
            {
                module_id: dto.moduleId,
                config_schema_version: dto.configSchemaVersion,
                config: dto.config,
            },
        );

        if (!Array.isArray(payload.errors)) {
            throw new ServiceUnavailableException('bot_platform_validation_payload_invalid');
        }

        return payload;
    }

    async listInstances(): Promise<AdminBotInstanceSummary[]> {
        const payload = await this.getJson<BotPlatformInstanceListResponse>('/admin/bot-instances');

        if (!Array.isArray(payload.instances)) {
            throw new ServiceUnavailableException('bot_platform_instances_payload_invalid');
        }

        return payload.instances;
    }

    async createInstance(dto: CreateBotInstanceDto): Promise<BotInstanceActionResult> {
        return await this.postJson<BotInstanceActionResult>('/admin/bot-instances', {
            instance_id: dto.instanceId,
            module_id: dto.moduleId,
            name: dto.name,
            mode: dto.mode,
            symbols: dto.symbols,
            timeframes: dto.timeframes,
            config_schema_version: dto.configSchemaVersion,
            config: dto.config,
        });
    }

    async enableInstance(instanceId: string): Promise<BotInstanceActionResult> {
        return await this.postJson<BotInstanceActionResult>(
            `/admin/bot-instances/${encodeURIComponent(instanceId)}/enable`,
            {},
        );
    }

    async pauseInstance(instanceId: string): Promise<BotInstanceActionResult> {
        return await this.postJson<BotInstanceActionResult>(
            `/admin/bot-instances/${encodeURIComponent(instanceId)}/pause`,
            {},
        );
    }

    async runInstance(instanceId: string, dto: RunBotInstanceDto = {}): Promise<ManualBotRunResult> {
        return await this.postJson<ManualBotRunResult>(
            `/admin/bot-instances/${encodeURIComponent(instanceId)}/run`,
            {
                idempotency_key: dto.idempotencyKey,
                correlation_id: dto.correlationId,
            },
        );
    }

    private async getJson<T>(path: string): Promise<T> {
        const url = `${this.getBaseUrl()}${path}`;
        let response: Response;

        try {
            response = await fetch(url, {
                method: 'GET',
                headers: {
                    accept: 'application/json',
                },
            });
        } catch {
            throw new ServiceUnavailableException('bot_platform_unavailable');
        }

        if (response.status === 404) {
            throw new NotFoundException('bot_module_not_found');
        }

        if (!response.ok) {
            throw new ServiceUnavailableException('bot_platform_unavailable');
        }

        return (await response.json()) as T;
    }

    private async postJson<T>(path: string, body: Record<string, unknown>): Promise<T> {
        const url = `${this.getBaseUrl()}${path}`;
        let response: Response;

        try {
            response = await fetch(url, {
                method: 'POST',
                headers: {
                    accept: 'application/json',
                    'content-type': 'application/json',
                },
                body: JSON.stringify(body),
            });
        } catch {
            throw new ServiceUnavailableException('bot_platform_unavailable');
        }

        if (response.status === 409) {
            return (await response.json()) as T;
        }

        if (!response.ok) {
            throw new ServiceUnavailableException('bot_platform_unavailable');
        }

        return (await response.json()) as T;
    }

    private getBaseUrl(): string {
        const configuredUrl =
            this.configService.get<string>('BOT_PLATFORM_BASE_URL') ?? DEFAULT_BOT_PLATFORM_BASE_URL;

        return configuredUrl.replace(/\/+$/, '');
    }
}
