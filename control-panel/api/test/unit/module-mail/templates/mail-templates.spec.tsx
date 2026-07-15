import { MailRenderService } from '@/module-mail/services/mail-render.service';
import { Test, TestingModule } from '@nestjs/testing';
import * as React from 'react';
import { VerificationCodeTemplate } from '@/module-mail/templates/verification-code.template';
import { PasswordResetTemplate } from '@/module-mail/templates/password-reset.template';
import { SecurityAlertTemplate } from '@/module-mail/templates/security-alert.template';
import { EmailChangeConfirmationTemplate } from '@/module-mail/templates/email-change-confirmation.template';

describe('MailTemplatesRendering', () => {
    let service: MailRenderService;

    beforeEach(async () => {
        const module: TestingModule = await Test.createTestingModule({
            providers: [MailRenderService],
        }).compile();

        service = module.get<MailRenderService>(MailRenderService);
    });

    it('should render VerificationCodeTemplate', async () => {
        const component = React.createElement(VerificationCodeTemplate, {
            userName: 'John Doe',
            code: '123456',
            expiresInMinutes: 15,
        });

        const html = await service.renderHtml(component);
        const text = await service.renderText(component);

        expect(html).toContain('John Doe');
        expect(html).toContain('123456');
        expect(html).toContain('Pampilo');

        expect(text).toContain('John Doe');
        expect(text).toContain('123456');
        expect(text).toContain('Pampilo');
    });

    it('should render PasswordResetTemplate', async () => {
        const component = React.createElement(PasswordResetTemplate, {
            userName: 'Jane Doe',
            resetLink: 'https://localhost/reset?token=abc',
            expiresInMinutes: 30,
        });

        const html = await service.renderHtml(component);
        const text = await service.renderText(component);

        expect(html).toContain('Jane Doe');
        expect(html).toContain('https://localhost/reset?token=abc');

        expect(text).toContain('Jane Doe');
        expect(text).toContain('https://localhost/reset?token=abc');
    });

    it('should render SecurityAlertTemplate', async () => {
        const component = React.createElement(SecurityAlertTemplate, {
            userName: 'Alert User',
            alertType: 'Suspicious Login',
            details: 'Login from an unrecognized browser/device.',
            timestamp: '2026-06-19T10:00:00Z',
        });

        const html = await service.renderHtml(component);
        const text = await service.renderText(component);

        expect(html).toContain('Alert User');
        expect(html).toContain('Suspicious Login');
        expect(html).toContain('Login from an unrecognized browser/device.');
        expect(html).toContain('2026-06-19T10:00:00Z');

        expect(text).toContain('Alert User');
        expect(text).toContain('Suspicious Login');
        expect(text).toContain('Login from an unrecognized browser/device.');
        expect(text).toContain('2026-06-19T10:00:00Z');
    });

    it('should render EmailChangeConfirmationTemplate', async () => {
        const component = React.createElement(EmailChangeConfirmationTemplate, {
            userName: 'Change User',
            newEmail: 'new-email@example.local',
            code: '654321',
            expiresInMinutes: 15,
        });

        const html = await service.renderHtml(component);
        const text = await service.renderText(component);

        expect(html).toContain('Change User');
        expect(html).toContain('new-email@example.local');
        expect(html).toContain('654321');
        expect(html).toContain('15');

        expect(text).toContain('Change User');
        expect(text).toContain('new-email@example.local');
        expect(text).toContain('654321');
        expect(text).toContain('15');
    });
});
