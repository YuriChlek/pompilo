import { NotFoundException, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { AdminBotsService } from '@/module-admin-bots/services/admin-bots.service';

describe('AdminBotsService', () => {
    let service: AdminBotsService;
    let configService: jest.Mocked<ConfigService>;
    let fetchMock: jest.Mock;

    beforeEach(() => {
        configService = {
            get: jest.fn((key: string) => {
                if (key === 'BOT_PLATFORM_BASE_URL') {
                    return 'http://bot-platform.local/';
                }
                return undefined;
            }),
        } as unknown as jest.Mocked<ConfigService>;
        fetchMock = jest.fn();
        global.fetch = fetchMock;
        service = new AdminBotsService(configService);
    });

    it('lists persisted modules through the bot-platform metadata API', async () => {
        const modules = [
            {
                module_id: 'spot_grid',
                display_name: 'Spot Grid',
                version: '0.1.0',
                status: 'ACTIVE',
                supported_modes: ['signal_only'],
                required_timeframes: ['1h'],
                required_market_data: ['snapshots'],
                supports_multi_symbol: false,
                config_schema_version: 1,
                config_schema_available: true,
            },
            {
                module_id: 'spot_greenwich',
                display_name: 'Spot Greenwich',
                version: '0.1.0',
                status: 'ACTIVE',
                supported_modes: ['signal_only'],
                required_timeframes: ['1h'],
                required_market_data: ['snapshots'],
                supports_multi_symbol: false,
                config_schema_version: 1,
                config_schema_available: true,
            },
        ];
        fetchMock.mockResolvedValue(responseJson(200, { modules }));

        await expect(service.listModules()).resolves.toEqual(modules);
        expect(fetchMock).toHaveBeenCalledWith('http://bot-platform.local/admin/bot-modules', {
            method: 'GET',
            headers: {
                accept: 'application/json',
            },
        });
    });

    it('loads one config schema through the bot-platform metadata API', async () => {
        const schema = {
            module_id: 'spot_grid',
            config_schema_version: 1,
            config_schema: {
                schema_version: 1,
                sections: [],
            },
        };
        fetchMock.mockResolvedValue(responseJson(200, schema));

        await expect(service.getConfigSchema('spot_grid')).resolves.toEqual(schema);
        expect(fetchMock).toHaveBeenCalledWith(
            'http://bot-platform.local/admin/bot-modules/spot_grid/config-schema',
            expect.objectContaining({ method: 'GET' }),
        );
    });

    it('validates config through the bot-platform validation API', async () => {
        const result = {
            valid: false,
            errors: [{ field_path: 'risk.max_position_fraction', code: 'max_value', message: 'too high' }],
        };
        fetchMock.mockResolvedValue(responseJson(200, result));

        await expect(
            service.validateConfig({
                moduleId: 'spot_grid',
                configSchemaVersion: 1,
                config: { risk: { max_position_fraction: '2.0' } },
            }),
        ).resolves.toEqual(result);
        expect(fetchMock).toHaveBeenCalledWith(
            'http://bot-platform.local/admin/bot-instances/validate-config',
            expect.objectContaining({
                method: 'POST',
                body: JSON.stringify({
                    module_id: 'spot_grid',
                    config_schema_version: 1,
                    config: { risk: { max_position_fraction: '2.0' } },
                }),
            }),
        );
    });

    it('lists instances through the bot-platform lifecycle API', async () => {
        const instances = [{ instance_id: 'instance-1', module_id: 'spot_grid', status: 'ENABLED' }];
        fetchMock.mockResolvedValue(responseJson(200, { instances }));

        await expect(service.listInstances()).resolves.toEqual(instances);
        expect(fetchMock).toHaveBeenCalledWith('http://bot-platform.local/admin/bot-instances', {
            method: 'GET',
            headers: {
                accept: 'application/json',
            },
        });
    });

    it('creates instances through the bot-platform lifecycle API', async () => {
        const result = { accepted: true, instance_id: 'instance-1', status: 'CREATED', error_code: null };
        fetchMock.mockResolvedValue(responseJson(201, result));

        await expect(
            service.createInstance({
                moduleId: 'spot_grid',
                name: 'Grid ETH',
                mode: 'signal_only',
                symbols: ['ETHUSDT'],
                timeframes: ['1h'],
                configSchemaVersion: 1,
                config: { symbols: ['ETHUSDT'] },
            }),
        ).resolves.toEqual(result);
        expect(fetchMock).toHaveBeenCalledWith(
            'http://bot-platform.local/admin/bot-instances',
            expect.objectContaining({
                method: 'POST',
                body: JSON.stringify({
                    module_id: 'spot_grid',
                    name: 'Grid ETH',
                    mode: 'signal_only',
                    symbols: ['ETHUSDT'],
                    timeframes: ['1h'],
                    config_schema_version: 1,
                    config: { symbols: ['ETHUSDT'] },
                }),
            }),
        );
    });

    it('proxies enable pause and manual run actions through bot-platform', async () => {
        fetchMock
            .mockResolvedValueOnce(responseJson(200, { accepted: true, instance_id: 'instance-1', status: 'ENABLED', error_code: null }))
            .mockResolvedValueOnce(responseJson(200, { accepted: true, instance_id: 'instance-1', status: 'PAUSED', error_code: null }))
            .mockResolvedValueOnce(responseJson(202, { accepted: true, instance_id: 'instance-1', run_id: 'run-1', status: 'COMPLETE', error_code: null, duplicate: false }));

        await service.enableInstance('instance-1');
        await service.pauseInstance('instance-1');
        await service.runInstance('instance-1');

        expect(fetchMock).toHaveBeenNthCalledWith(
            1,
            'http://bot-platform.local/admin/bot-instances/instance-1/enable',
            expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
        );
        expect(fetchMock).toHaveBeenNthCalledWith(
            2,
            'http://bot-platform.local/admin/bot-instances/instance-1/pause',
            expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
        );
        expect(fetchMock).toHaveBeenNthCalledWith(
            3,
            'http://bot-platform.local/admin/bot-instances/instance-1/run',
            expect.objectContaining({ method: 'POST', body: JSON.stringify({}) }),
        );
    });

    it('passes structured lifecycle rejections through to the UI facade', async () => {
        const rejection = {
            accepted: false,
            instance_id: 'instance-1',
            status: 'PAUSED',
            error_code: 'bot_instance_not_enabled',
            duplicate: false,
        };
        fetchMock.mockResolvedValue(responseJson(409, rejection));

        await expect(service.runInstance('instance-1')).resolves.toEqual(rejection);
    });

    it('passes manual run idempotency metadata to bot-platform', async () => {
        fetchMock.mockResolvedValue(
            responseJson(202, {
                accepted: true,
                instance_id: 'instance-1',
                run_id: 'run-1',
                status: 'COMPLETE',
                error_code: null,
                duplicate: false,
            }),
        );

        await service.runInstance('instance-1', {
            idempotencyKey: 'stage26-full-docker-smoke-v1',
            correlationId: 'stage26-full-docker-smoke',
        });

        expect(fetchMock).toHaveBeenCalledWith(
            'http://bot-platform.local/admin/bot-instances/instance-1/run',
            expect.objectContaining({
                method: 'POST',
                body: JSON.stringify({
                    idempotency_key: 'stage26-full-docker-smoke-v1',
                    correlation_id: 'stage26-full-docker-smoke',
                }),
            }),
        );
    });

    it('maps upstream 404 to a module not found error', async () => {
        fetchMock.mockResolvedValue(responseJson(404, { error: 'module_not_found' }));

        await expect(service.getConfigSchema('missing')).rejects.toThrow(NotFoundException);
    });

    it('does not expose direct DB or lifecycle behavior when bot-platform is unavailable', async () => {
        fetchMock.mockRejectedValue(new Error('connection refused'));

        await expect(service.listModules()).rejects.toThrow(ServiceUnavailableException);
    });
});

function responseJson(status: number, payload: unknown): Response {
    return {
        ok: status >= 200 && status < 300,
        status,
        json: jest.fn().mockResolvedValue(payload),
    } as unknown as Response;
}
