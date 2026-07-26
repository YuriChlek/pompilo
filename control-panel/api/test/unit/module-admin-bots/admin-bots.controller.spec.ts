import { Test, TestingModule } from '@nestjs/testing';
import { ROLES_KEY } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { AdminBotsController } from '@/module-admin-bots/controllers/admin-bots.controller';
import { AdminBotsService } from '@/module-admin-bots/services/admin-bots.service';

describe('AdminBotsController', () => {
    let controller: AdminBotsController;
    let adminBotsService: {
        listModules: jest.Mock;
        getConfigSchema: jest.Mock;
        validateConfig: jest.Mock;
        listInstances: jest.Mock;
        createInstance: jest.Mock;
        enableInstance: jest.Mock;
        pauseInstance: jest.Mock;
        runInstance: jest.Mock;
    };

    beforeEach(async () => {
        adminBotsService = {
            listModules: jest.fn(),
            getConfigSchema: jest.fn(),
            validateConfig: jest.fn(),
            listInstances: jest.fn(),
            createInstance: jest.fn(),
            enableInstance: jest.fn(),
            pauseInstance: jest.fn(),
            runInstance: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [AdminBotsController],
            providers: [{ provide: AdminBotsService, useValue: adminBotsService }],
        }).compile();

        controller = module.get<AdminBotsController>(AdminBotsController);
    });

    it('is restricted to platform admin roles', () => {
        const roles = Reflect.getMetadata(ROLES_KEY, AdminBotsController) as
            | UserRoles[]
            | undefined;

        expect(roles).toEqual([UserRoles.PLATFORM_ADMIN, UserRoles.SUPER_ADMIN]);
    });

    it('returns read-only module list metadata', async () => {
        const modules = [{ module_id: 'spot_grid' }];
        adminBotsService.listModules.mockResolvedValue(modules);

        await expect(controller.listModules()).resolves.toEqual(modules);
        expect(adminBotsService.listModules).toHaveBeenCalledTimes(1);
    });

    it('returns read-only config schema metadata', async () => {
        const schema = { module_id: 'spot_greenwich', config_schema: null };
        adminBotsService.getConfigSchema.mockResolvedValue(schema);

        await expect(controller.getConfigSchema('spot_greenwich')).resolves.toEqual(schema);
        expect(adminBotsService.getConfigSchema).toHaveBeenCalledWith('spot_greenwich');
    });

    it('proxies config validation without lifecycle actions', async () => {
        const result = { valid: true, errors: [] };
        const dto = { moduleId: 'spot_grid', configSchemaVersion: 1, config: {} };
        adminBotsService.validateConfig.mockResolvedValue(result);

        await expect(controller.validateConfig(dto)).resolves.toEqual(result);
        expect(adminBotsService.validateConfig).toHaveBeenCalledWith(dto);
    });

    it('proxies instance lifecycle actions through the service facade', async () => {
        const createDto = {
            moduleId: 'spot_grid',
            mode: 'signal_only',
            symbols: ['ETHUSDT'],
            timeframes: ['1h'],
            configSchemaVersion: 1,
            config: {},
        };
        adminBotsService.listInstances.mockResolvedValue([{ instance_id: 'instance-1' }]);
        adminBotsService.createInstance.mockResolvedValue({ accepted: true });
        adminBotsService.enableInstance.mockResolvedValue({ accepted: true });
        adminBotsService.pauseInstance.mockResolvedValue({ accepted: true });
        adminBotsService.runInstance.mockResolvedValue({ accepted: true, run_id: 'run-1' });

        await expect(controller.listInstances()).resolves.toEqual([{ instance_id: 'instance-1' }]);
        await expect(controller.createInstance(createDto)).resolves.toEqual({ accepted: true });
        await expect(controller.enableInstance('instance-1')).resolves.toEqual({ accepted: true });
        await expect(controller.pauseInstance('instance-1')).resolves.toEqual({ accepted: true });
        await expect(controller.runInstance('instance-1', {})).resolves.toEqual({ accepted: true, run_id: 'run-1' });

        expect(adminBotsService.createInstance).toHaveBeenCalledWith(createDto);
        expect(adminBotsService.enableInstance).toHaveBeenCalledWith('instance-1');
        expect(adminBotsService.pauseInstance).toHaveBeenCalledWith('instance-1');
        expect(adminBotsService.runInstance).toHaveBeenCalledWith('instance-1', {});
    });
});
