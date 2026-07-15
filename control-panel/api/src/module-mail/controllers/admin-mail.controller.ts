import {
    Body,
    Controller,
    Get,
    HttpCode,
    HttpStatus,
    Patch,
    Post,
    Req,
    UseGuards,
    BadRequestException,
} from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { Authorisation } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { SendTestEmailDto, UpdateMailSettingsDto } from '@/module-mail/dto/admin-mail.dto';
import type { Request } from 'express';
import { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { AdminMailActionRateLimit } from '@/module-mail/decorators/admin-mail-action-rate-limit.decorator';
import { AdminMailActionRateLimitGuard } from '@/module-mail/guards/admin-mail-action-rate-limit.guard';

@ApiTags('Admin / Mail')
@ApiBearerAuth()
@Controller('admin/mail')
@Authorisation(UserRoles.PLATFORM_ADMIN, UserRoles.SUPER_ADMIN)
export class AdminMailController {
    constructor(private readonly mailSettingsService: MailSettingsService) {}

    @Get('settings')
    @ApiOperation({ summary: 'Get current mail settings (secrets masked)' })
    async getSettings() {
        return await this.mailSettingsService.getSettings();
    }

    @Get('setup-state')
    @ApiOperation({ summary: 'Get mail setup state' })
    async getSetupState() {
        return {
            state: await this.mailSettingsService.getSetupState(),
        };
    }

    @Patch('settings')
    @ApiOperation({ summary: 'Update mail settings' })
    async updateSettings(@Body() dto: UpdateMailSettingsDto, @Req() req: Request) {
        const user = req.user as AccessTokenPayload;
        const userId = user.userId;
        try {
            return await this.mailSettingsService.updateSettings(dto, userId);
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            throw new BadRequestException(message);
        }
    }

    @Post('send-test-email')
    @HttpCode(HttpStatus.OK)
    @UseGuards(AdminMailActionRateLimitGuard)
    @AdminMailActionRateLimit({ action: 'test_email' })
    @ApiOperation({ summary: 'Send a test email using a template' })
    async sendTestEmail(@Body() dto: SendTestEmailDto, @Req() req: Request) {
        const user = req.user as AccessTokenPayload;
        try {
            return await this.mailSettingsService.sendTestEmail(
                dto.to,
                dto.templateId,
                user.userId,
                user.email,
            );
        } catch (error: unknown) {
            const message = error instanceof Error ? error.message : String(error);
            throw new BadRequestException(message);
        }
    }
}
