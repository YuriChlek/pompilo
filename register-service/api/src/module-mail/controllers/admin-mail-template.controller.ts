import { Controller, Get, Param } from '@nestjs/common';
import { ApiBearerAuth, ApiOperation, ApiTags } from '@nestjs/swagger';
import { Authorisation } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { MailTemplatePreviewService } from '../services/mail-template-preview.service';
import {
    MailTemplatePreview,
    MailTemplateSummary,
} from '../interfaces/mail-template-preview.interfaces';

@ApiTags('Admin / Mail')
@ApiBearerAuth()
@Controller('admin/mail/templates')
@Authorisation(UserRoles.PLATFORM_ADMIN, UserRoles.SUPER_ADMIN)
export class AdminMailTemplateController {
    constructor(private readonly mailTemplatePreviewService: MailTemplatePreviewService) {}

    @Get()
    @ApiOperation({ summary: 'Get all available mail templates summaries for preview' })
    getTemplates(): MailTemplateSummary[] {
        return this.mailTemplatePreviewService.getTemplateSummaries();
    }

    @Get(':templateId/preview')
    @ApiOperation({ summary: 'Get preview for a specific mail template' })
    getTemplatePreview(@Param('templateId') templateId: string): Promise<MailTemplatePreview> {
        return this.mailTemplatePreviewService.getTemplatePreview(templateId);
    }
}
