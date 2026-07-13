import { Injectable, NotFoundException } from '@nestjs/common';
import * as React from 'react';
import { MailRenderService } from './mail-render.service';
import { MAIL_TEMPLATE_PREVIEW_REGISTRY } from '../constants/mail-template-preview.constants';
import {
    MailTemplatePreview,
    MailTemplateSummary,
} from '../interfaces/mail-template-preview.interfaces';

@Injectable()
export class MailTemplatePreviewService {
    constructor(private readonly mailRenderService: MailRenderService) {}

    getTemplateSummaries(): MailTemplateSummary[] {
        return Object.values(MAIL_TEMPLATE_PREVIEW_REGISTRY).map(item => ({
            id: item.id,
            name: item.name,
            description: item.description,
            subject: item.subject,
        }));
    }

    async getTemplatePreview(templateId: string): Promise<MailTemplatePreview> {
        const item = MAIL_TEMPLATE_PREVIEW_REGISTRY[templateId];
        if (!item) {
            throw new NotFoundException(`Mail template with ID "${templateId}" not found`);
        }

        const component = React.createElement(item.component, item.demoProps);

        const html = await this.mailRenderService.renderHtml(component);
        const text = await this.mailRenderService.renderText(component);

        return {
            id: item.id,
            name: item.name,
            description: item.description,
            subject: item.subject,
            html,
            text,
        };
    }
}
