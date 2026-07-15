import { describe, expect, it, vi, beforeEach } from 'vitest';
import { adminBotsApiService } from '@/features/module-admin-bots/api-service/client';
import { apiClient } from '@/lib/http-client/http-client';

vi.mock('@/lib/http-client/http-client', () => ({
    apiClient: {
        get: vi.fn(),
        post: vi.fn(),
    },
}));

describe('adminBotsApiService', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('lists modules through the generic admin metadata endpoint', async () => {
        const modules = [
            {
                module_id: 'fixture_module',
                display_name: 'Fixture Module',
                version: '1.0.0',
                status: 'ACTIVE',
                supported_modes: ['dry_run'],
                required_timeframes: ['1h'],
                required_market_data: ['snapshots'],
                supports_multi_symbol: true,
                config_schema_version: 1,
                config_schema_available: true,
            },
        ];
        vi.mocked(apiClient.get).mockResolvedValue({
            success: true,
            statusCode: 200,
            timestamp: '2026-01-01T00:00:00.000Z',
            data: modules,
        });

        const result = await adminBotsApiService.getModules();

        expect(apiClient.get).toHaveBeenCalledWith('/admin/bot-modules');
        expect(result).toEqual(modules);
    });

    it('loads config schema through module metadata', async () => {
        const schema = {
            module_id: 'fixture_module',
            config_schema_version: 1,
            config_schema: {
                schema_version: 1,
                sections: [],
            },
        };
        vi.mocked(apiClient.get).mockResolvedValue({
            success: true,
            statusCode: 200,
            timestamp: '2026-01-01T00:00:00.000Z',
            data: schema,
        });

        const result = await adminBotsApiService.getConfigSchema('fixture_module');

        expect(apiClient.get).toHaveBeenCalledWith('/admin/bot-modules/fixture_module/config-schema');
        expect(result).toEqual(schema);
    });

    it('submits config payloads to backend validation', async () => {
        vi.mocked(apiClient.post).mockResolvedValue({
            success: true,
            statusCode: 200,
            timestamp: '2026-01-01T00:00:00.000Z',
            data: { valid: false, errors: ['risk is required'] },
        });

        const dto = {
            moduleId: 'fixture_module',
            configSchemaVersion: 1,
            config: { risk: 'low' },
        };
        const result = await adminBotsApiService.validateConfig(dto);

        expect(apiClient.post).toHaveBeenCalledWith('/admin/bot-instances/validate-config', dto);
        expect(result).toEqual({ valid: false, errors: ['risk is required'] });
    });
});
