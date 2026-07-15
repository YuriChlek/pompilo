import { Logger } from '@nestjs/common';
import { Test, TestingModule } from '@nestjs/testing';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { MailRenderService } from '@/module-mail/services/mail-render.service';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { MAIL_SERVICE } from '@/module-mail/constants/mail.constants';
import type { RepositoryTransaction } from '@/module-drizzle/repository/transaction.repository';

describe('MailTemplateService', () => {
    let service: MailTemplateService;
    let mailService: { createDeliveryRequest: jest.Mock };
    let mailRenderService: { renderHtml: jest.Mock; renderText: jest.Mock };
    let mailSettingsService: { getCachedSettings: jest.Mock };

    beforeEach(async () => {
        mailService = { createDeliveryRequest: jest.fn() };
        mailRenderService = {
            renderHtml: jest.fn().mockResolvedValue('<html></html>'),
            renderText: jest.fn().mockResolvedValue('text content'),
        };

        mailSettingsService = {
            getCachedSettings: jest.fn().mockResolvedValue({
                clientPublicUrl: 'https://localhost',
            }),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailTemplateService,
                { provide: MAIL_SERVICE, useValue: mailService },
                { provide: MailRenderService, useValue: mailRenderService },
                { provide: MailSettingsService, useValue: mailSettingsService },
            ],
        }).compile();

        service = module.get<MailTemplateService>(MailTemplateService);
    });

    it('should send verification code', async () => {
        await service.sendVerificationCode('test@test.com', 'John', '123456');

        expect(mailRenderService.renderHtml).toHaveBeenCalled();
        expect(mailRenderService.renderText).toHaveBeenCalled();
        expect(mailService.createDeliveryRequest).toHaveBeenCalledWith(
            expect.objectContaining({
                to: 'test@test.com',
                subject: 'Your verification code',
                html: '<html></html>',
                text: 'text content',
            }),
            undefined,
        );
    });

    it('should send registration email verification through the outbox service', async () => {
        await service.sendEmailVerification(
            'test@test.com',
            'John',
            'https://localhost/auth/verify-email?token=token',
        );

        expect(mailService.createDeliveryRequest).toHaveBeenCalledWith(
            expect.objectContaining({
                to: 'test@test.com',
                subject: 'Verify your email address',
            }),
            undefined,
        );
    });

    it('should send verification code through strict method', async () => {
        mailService.createDeliveryRequest.mockResolvedValue({
            outboxId: 'outbox-1',
            idempotencyKey: 'key-1',
            status: 'accepted',
        });

        await expect(
            service.sendVerificationCodeOrThrow('test@test.com', 'John', '123456'),
        ).resolves.toEqual({
            outboxId: 'outbox-1',
            idempotencyKey: 'key-1',
            status: 'accepted',
        });
    });

    it('should throw from strict method when verification code delivery request fails', async () => {
        mailService.createDeliveryRequest.mockRejectedValue(new Error('outbox unavailable'));

        await expect(
            service.sendVerificationCodeOrThrow('test@test.com', 'John', '123456'),
        ).rejects.toThrow('Verification email could not be queued.');
    });

    it('should send password reset link', async () => {
        const link = 'https://localhost/reset';
        await service.sendPasswordReset('test@test.com', 'John', link);

        expect(mailService.createDeliveryRequest).toHaveBeenCalledWith(
            expect.objectContaining({
                to: 'test@test.com',
                subject: 'Reset your password',
            }),
            undefined,
        );
    });

    it('should pass transaction client if provided', async () => {
        const tx = { some: 'tx' } as unknown as RepositoryTransaction;
        await service.sendVerificationCode('test@test.com', 'John', '123456', 15, tx);

        expect(mailService.createDeliveryRequest).toHaveBeenCalledWith(expect.any(Object), tx);
    });

    it('should send security alert', async () => {
        await service.sendSecurityAlert(
            'test@test.com',
            'John',
            'Suspicious Activity',
            'Details info',
        );

        expect(mailSettingsService.getCachedSettings).toHaveBeenCalled();
        expect(mailService.createDeliveryRequest).toHaveBeenCalledWith(
            expect.objectContaining({
                to: 'test@test.com',
                subject: 'Security Alert',
            }),
            undefined,
        );
        expect(mailRenderService.renderHtml).toHaveBeenCalledWith(
            expect.objectContaining({
                props: expect.objectContaining({
                    reviewDevicesLink: 'https://localhost/account/security',
                }) as unknown,
            }),
        );
    });

    it('should redact recipient addresses from template delivery error logs', async () => {
        const logSpy = jest.spyOn(Logger.prototype, 'error').mockImplementation(() => undefined);
        mailService.createDeliveryRequest.mockRejectedValue(new Error('outbox unavailable'));

        await service.sendPasswordReset(
            'sensitive.user@example.com',
            'User',
            'https://localhost/auth/reset-password?token=secret',
        );

        const logged = logSpy.mock.calls.flat().join(' ');
        expect(logged).toContain('s***@e***.com');
        expect(logged).not.toContain('sensitive.user@example.com');
        logSpy.mockRestore();
    });
});
