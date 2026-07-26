import { Test, TestingModule } from '@nestjs/testing';
import { AdminMailController } from '@/module-mail/controllers/admin-mail.controller';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY } from '@/module-mail/constants/admin-mail-action-rate-limit.constants';
import { RedisService } from '@/common/redis/redis.service';
import type { Request } from 'express';
import { METHOD_METADATA, PATH_METADATA } from '@nestjs/common/constants';
import { RequestMethod, BadRequestException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ROLES_KEY } from '@/module-auth/decorators/auth.decorator';
import { UserRoles } from '@/module-auth/enums/auth.enums';

describe('AdminMailController', () => {
    let controller: AdminMailController;
    let mailSettingsService: {
        getSettings: jest.Mock;
        getSetupState: jest.Mock;
        updateSettings: jest.Mock;
        sendTestEmail: jest.Mock;
    };

    const adminUser = {
        userId: 'admin-1',
        email: 'admin@example.com',
    } as AccessTokenPayload;

    const getHandler = (methodName: keyof AdminMailController): ((...args: unknown[]) => unknown) =>
        Object.getOwnPropertyDescriptor(AdminMailController.prototype, methodName)!.value as (
            ...args: unknown[]
        ) => unknown;

    beforeEach(async () => {
        mailSettingsService = {
            getSettings: jest.fn(),
            getSetupState: jest.fn(),
            updateSettings: jest.fn(),
            sendTestEmail: jest.fn(),
        };

        const module: TestingModule = await Test.createTestingModule({
            controllers: [AdminMailController],
            providers: [
                { provide: MailSettingsService, useValue: mailSettingsService },
                { provide: RedisTokenService, useValue: {} }, // Mock for guards
                {
                    provide: RedisService,
                    useValue: {
                        getClient: () => ({
                            eval: jest.fn(),
                            pttl: jest.fn(),
                        }),
                    },
                },
                {
                    provide: ConfigService,
                    useValue: {
                        get: jest.fn().mockReturnValue('development'),
                    },
                },
            ],
        }).compile();

        controller = module.get<AdminMailController>(AdminMailController);
    });

    it('should be defined', () => {
        expect(controller).toBeDefined();
    });

    it('should allow the neutral platform admin role during compatibility migration', () => {
        const roles = Reflect.getMetadata(ROLES_KEY, AdminMailController) as
            | UserRoles[]
            | undefined;

        expect(roles).toContain(UserRoles.PLATFORM_ADMIN);
        expect(roles).toContain(UserRoles.SUPER_ADMIN);
        expect(roles).toContain(UserRoles.PLATFORM_ADMIN);
    });

    it('should get settings', async () => {
        mailSettingsService.getSettings.mockResolvedValue({ smtpHost: 'test' });
        const result = await controller.getSettings();
        expect(result).toEqual({ smtpHost: 'test' });
    });

    it('should update settings', async () => {
        const dto = { smtpHost: 'new-host' };
        const req = {
            user: adminUser,
        } as unknown as Request;
        mailSettingsService.updateSettings.mockResolvedValue({ smtpHost: 'new-host' });

        const result = await controller.updateSettings(dto, req);
        expect(mailSettingsService.updateSettings).toHaveBeenCalledWith(dto, 'admin-1');
        expect(result).toEqual({ smtpHost: 'new-host' });
    });

    it('should send test email to requested recipient address', async () => {
        const req = { user: adminUser } as unknown as Request;
        mailSettingsService.sendTestEmail.mockResolvedValue({ outboxId: '1' });

        const result = await controller.sendTestEmail(
            { to: 'recipient@example.com', templateId: 'security-alert' },
            req,
        );
        expect(mailSettingsService.sendTestEmail).toHaveBeenCalledWith(
            'recipient@example.com',
            'security-alert',
            'admin-1',
            'admin@example.com',
        );
        expect(result).toEqual({ outboxId: '1' });
    });

    it('should throw BadRequestException when sendTestEmail service throws', async () => {
        const req = { user: adminUser } as unknown as Request;
        mailSettingsService.sendTestEmail.mockRejectedValue(new Error('Template not found'));

        let error: unknown;
        try {
            await controller.sendTestEmail(
                { to: 'recipient@example.com', templateId: 'unknown' },
                req,
            );
        } catch (e: unknown) {
            error = e;
        }

        expect(error).toBeInstanceOf(BadRequestException);
        expect((error as BadRequestException).message).toBe('Template not found');
    });

    it('should expose the test email action as POST /send-test-email', () => {
        const handler = getHandler('sendTestEmail');

        expect(Reflect.getMetadata(PATH_METADATA, handler)).toBe('send-test-email');
        expect(Reflect.getMetadata(METHOD_METADATA, handler)).toBe(RequestMethod.POST);
    });

    it('should protect the test email endpoint with admin mail rate limits', () => {
        expect(
            Reflect.getMetadata(
                ADMIN_MAIL_ACTION_RATE_LIMIT_METADATA_KEY,
                getHandler('sendTestEmail'),
            ),
        ).toEqual({ action: 'test_email' });
    });
});
