import { Body, Controller, Get, Param, Post } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { Authorisation } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { AdminBotsService } from '@/module-admin-bots/services/admin-bots.service';
import type {
    CreateBotInstanceDto,
    RunBotInstanceDto,
    ValidateBotConfigDto,
} from '@/module-admin-bots/interfaces/admin-bots.interfaces';

@ApiTags('Admin / Bot Modules')
@ApiBearerAuth()
@Controller('admin')
@Authorisation(UserRoles.PLATFORM_ADMIN, UserRoles.SUPER_ADMIN)
export class AdminBotsController {
    constructor(private readonly adminBotsService: AdminBotsService) {}

    @Get('bot-modules')
    @ApiOperation({ summary: 'List read-only bot module metadata' })
    async listModules() {
        return await this.adminBotsService.listModules();
    }

    @Get('bot-modules/:moduleId/config-schema')
    @ApiOperation({ summary: 'Get persisted bot module config schema' })
    async getConfigSchema(@Param('moduleId') moduleId: string) {
        return await this.adminBotsService.getConfigSchema(moduleId);
    }

    @Post('bot-instances/validate-config')
    @ApiOperation({ summary: 'Validate bot instance config against persisted module schema' })
    async validateConfig(@Body() dto: ValidateBotConfigDto) {
        return await this.adminBotsService.validateConfig(dto);
    }

    @Get('bot-instances')
    @ApiOperation({ summary: 'List bot instances through bot-platform facade' })
    async listInstances() {
        return await this.adminBotsService.listInstances();
    }

    @Post('bot-instances')
    @ApiOperation({ summary: 'Create bot instance through bot-platform facade' })
    async createInstance(@Body() dto: CreateBotInstanceDto) {
        return await this.adminBotsService.createInstance(dto);
    }

    @Post('bot-instances/:instanceId/enable')
    @ApiOperation({ summary: 'Enable bot instance through bot-platform facade' })
    async enableInstance(@Param('instanceId') instanceId: string) {
        return await this.adminBotsService.enableInstance(instanceId);
    }

    @Post('bot-instances/:instanceId/pause')
    @ApiOperation({ summary: 'Pause bot instance through bot-platform facade' })
    async pauseInstance(@Param('instanceId') instanceId: string) {
        return await this.adminBotsService.pauseInstance(instanceId);
    }

    @Post('bot-instances/:instanceId/run')
    @ApiOperation({ summary: 'Run bot instance manually through bot-platform facade' })
    async runInstance(@Param('instanceId') instanceId: string, @Body() dto: RunBotInstanceDto) {
        return await this.adminBotsService.runInstance(instanceId, dto);
    }
}
