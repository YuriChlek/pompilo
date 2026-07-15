import { Test, TestingModule } from '@nestjs/testing';
import * as React from 'react';
import { MAIL_TEMPLATE_PREVIEW_REGISTRY } from '@/module-mail/constants/mail-template-preview.constants';
import { MailRenderService } from '@/module-mail/services/mail-render.service';

describe('MAIL_TEMPLATE_PREVIEW_REGISTRY', () => {
    let service: MailRenderService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [MailRenderService],
        }).compile();

        service = module.get<MailRenderService>(MailRenderService);
    });

    it('should contain exactly the 5 identity and security templates', () => {
        const keys = Object.keys(MAIL_TEMPLATE_PREVIEW_REGISTRY);
        expect(keys).toHaveLength(5);
        expect(keys).toContain('email-verification');
        expect(keys).toContain('verification-code');
        expect(keys).toContain('password-reset');
        expect(keys).toContain('security-alert');
        expect(keys).toContain('email-change-confirmation');
    });

    it('should have valid metadata for all templates', () => {
        for (const key of Object.keys(MAIL_TEMPLATE_PREVIEW_REGISTRY)) {
            const item = MAIL_TEMPLATE_PREVIEW_REGISTRY[key];
            expect(item.id).toBe(key);
            expect(item.name).toBeTruthy();
            expect(item.description).toBeTruthy();
            expect(item.subject).toBeTruthy();
            expect(item.component).toBeDefined();
            expect(item.demoProps).toBeDefined();
        }
    });

    it('should render all templates successfully with their demoProps', async () => {
        for (const key of Object.keys(MAIL_TEMPLATE_PREVIEW_REGISTRY)) {
            const item = MAIL_TEMPLATE_PREVIEW_REGISTRY[key];
            const component = React.createElement(item.component, item.demoProps);

            const html = await service.renderHtml(component);
            const text = await service.renderText(component);

            expect(html).toBeTruthy();
            expect(text).toBeTruthy();

            // Verify a basic layout or content piece
            expect(html).toContain('Pampilo');
        }
    });
});
