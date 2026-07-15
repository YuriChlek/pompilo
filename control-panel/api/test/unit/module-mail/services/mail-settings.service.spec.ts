import { Test, TestingModule } from '@nestjs/testing';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { MailSettingsRepository } from '@/module-mail/repository/mail-settings.repository';
import { MailAuditEventRepository } from '@/module-mail/repository/mail-audit-event.repository';
import { MailEncryptionService } from '@/module-mail/services/mail-encryption.service';
import { MailRedisService } from '@/module-mail/services/mail-redis.service';
import { MailRedisInvalidationService } from '@/module-mail/services/mail-redis-invalidation.service';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { MailRenderService } from '@/module-mail/services/mail-render.service';
import { MAIL_SERVICE } from '@/module-mail/constants/mail.constants';
import {
    RepositoryTransaction,
    TransactionRepository,
} from '@/module-drizzle/repository/transaction.repository';
import { MAIL_TEMPLATE_PREVIEW_REGISTRY } from '@/module-mail/constants/mail-template-preview.constants';

describe('MailSettingsService', () => {
    let service: MailSettingsService;
    let repository: {
        findSingleton: jest.Mock;
        create: jest.Mock;
        update: jest.Mock;
    };
    let auditRepository: {
        create: jest.Mock;
    };
    let encryptionService: {
        encryptMailSecret: jest.Mock;
        maskMailSecretPresence: jest.Mock;
    };
    let renderService: {
        renderHtml: jest.Mock;
        renderText: jest.Mock;
    };
    let redisService: {
        getFailureCount: jest.Mock;
        getDeliveryErrors: jest.Mock;
        getLastSuccessfulSendAt: jest.Mock;
    };
    let mailService: {
        createDeliveryRequest: jest.Mock;
        verifyTransport: jest.Mock;
    };
    let templateService: {
        sendSecurityAlert: jest.Mock;
    };
    let invalidationService: {
        invalidate: jest.Mock;
        onInvalidate: jest.Mock;
    };
    let transactionRepository: {
        run: jest.Mock;
    };
    const transaction = {} as RepositoryTransaction;

    beforeEach(async () => {
        repository = {
            findSingleton: jest.fn(),
            create: jest.fn(),
            update: jest.fn(),
        };
        auditRepository = {
            create: jest.fn(),
        };

        encryptionService = {
            encryptMailSecret: jest.fn(),
            maskMailSecretPresence: jest.fn(),
        };

        redisService = {
            getFailureCount: jest.fn(),
            getDeliveryErrors: jest.fn(),
            getLastSuccessfulSendAt: jest.fn(),
        };

        mailService = {
            createDeliveryRequest: jest.fn(),
            verifyTransport: jest.fn(),
        };

        renderService = {
            renderHtml: jest.fn().mockResolvedValue('<html>Demo HTML</html>'),
            renderText: jest.fn().mockResolvedValue('Demo Text'),
        };

        templateService = {
            sendSecurityAlert: jest.fn(),
        };

        invalidationService = {
            invalidate: jest.fn(),
            onInvalidate: jest.fn(),
        };

        transactionRepository = {
            run: jest
                .fn()
                .mockImplementation((work: (tx: RepositoryTransaction) => Promise<unknown>) =>
                    work(transaction),
                ),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                MailSettingsService,
                { provide: MailSettingsRepository, useValue: repository },
                { provide: MailAuditEventRepository, useValue: auditRepository },
                { provide: MailEncryptionService, useValue: encryptionService },
                { provide: MailRedisService, useValue: redisService },
                {
                    provide: MailRedisInvalidationService,
                    useValue: invalidationService,
                },
                { provide: MailTemplateService, useValue: templateService },
                { provide: MAIL_SERVICE, useValue: mailService },
                { provide: MailRenderService, useValue: renderService },
                { provide: TransactionRepository, useValue: transactionRepository },
            ],
        }).compile();

        service = module.get<MailSettingsService>(MailSettingsService);
    });

    describe('onApplicationBootstrap', () => {
        it('should register mail settings cache invalidation without creating settings', () => {
            service.onApplicationBootstrap();

            expect(invalidationService.onInvalidate).toHaveBeenCalledWith(expect.any(Function));
            expect(repository.create).not.toHaveBeenCalled();
            expect(repository.findSingleton).not.toHaveBeenCalled();
            expect(mailService.verifyTransport).not.toHaveBeenCalled();
        });
    });

    describe('getSetupState', () => {
        it('should return unconfigured if no settings exist', async () => {
            repository.findSingleton.mockResolvedValue(null);
            const state = await service.getSetupState();
            expect(state).toBe('unconfigured');
        });

        it('should return configured if settings exist', async () => {
            repository.findSingleton.mockResolvedValue({
                confirmedAt: null,
            });
            const state = await service.getSetupState();
            expect(state).toBe('configured');
        });
    });

    describe('updateSettings', () => {
        it('should save enabled settings without SMTP verification', async () => {
            repository.findSingleton.mockResolvedValue({ id: '1', smtpHost: 'old' });
            encryptionService.maskMailSecretPresence.mockImplementation((s: unknown) => s);

            const dto = { smtpHost: 'new', enabled: true };
            await service.updateSettings(dto, 'admin-1');

            expect(transactionRepository.run).toHaveBeenCalled();
            expect(mailService.verifyTransport).not.toHaveBeenCalled();
            expect(repository.update).toHaveBeenCalledWith(
                '1',
                expect.objectContaining({
                    smtpHost: 'new',
                    enabled: true,
                }),
                transaction,
            );
            expect(auditRepository.create).toHaveBeenCalledWith(
                {
                    action: 'mail_settings_updated',
                    adminUserId: 'admin-1',
                    payload: {
                        changedFields: ['enabled', 'smtpHost'],
                        enabled: true,
                    },
                },
                transaction,
            );
        });

        it('should save without verification when disabled', async () => {
            repository.findSingleton.mockResolvedValue({ id: '1' });
            encryptionService.maskMailSecretPresence.mockImplementation((s: unknown) => s);

            const dto = { smtpHost: 'new', enabled: false };
            await service.updateSettings(dto, 'admin-1');

            expect(mailService.verifyTransport).not.toHaveBeenCalled();
            expect(repository.update).toHaveBeenCalledWith(
                '1',
                expect.objectContaining({
                    smtpHost: 'new',
                    enabled: false,
                }),
                transaction,
            );
            const updateCall = repository.update.mock.calls[0] as [string, Record<string, unknown>];
            const updateData = updateCall[1];
            expect(updateData.lastVerifiedAt).toBeUndefined();
            expect(auditRepository.create).toHaveBeenCalledWith(
                {
                    action: 'mail_settings_disabled',
                    adminUserId: 'admin-1',
                    payload: {
                        changedFields: ['enabled', 'smtpHost'],
                        enabled: false,
                    },
                },
                transaction,
            );
        });

        it('should audit SMTP password changes without storing secret values', async () => {
            repository.findSingleton.mockResolvedValue({ id: '1', smtpPasswordEncrypted: 'old' });
            encryptionService.encryptMailSecret.mockReturnValue('encrypted-new-password');
            encryptionService.maskMailSecretPresence.mockImplementation((s: unknown) => s);

            await service.updateSettings(
                { smtpHost: 'new', smtpPassword: 'plain-secret', enabled: true },
                'admin-1',
            );

            expect(auditRepository.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    payload: {
                        changedFields: ['enabled', 'smtpHost'],
                        enabled: true,
                        hasSmtpPasswordChange: true,
                    },
                }),
                transaction,
            );
            expect(JSON.stringify(auditRepository.create.mock.calls)).not.toContain('plain-secret');
        });
    });

    describe('sendTestEmail', () => {
        it('should send test email to requested address and audit outbox id without body content', async () => {
            mailService.createDeliveryRequest.mockResolvedValue({ outboxId: 'outbox-1' });

            const result = await service.sendTestEmail(
                'recipient@example.com',
                'security-alert',
                'admin-1',
                'admin@example.com',
            );

            expect(result).toEqual({ outboxId: 'outbox-1' });
            expect(mailService.createDeliveryRequest).toHaveBeenCalledWith({
                to: 'recipient@example.com',
                subject: 'Security Notification',
                html: '<html>Demo HTML</html>',
                text: 'Demo Text',
            });
            expect(renderService.renderHtml).toHaveBeenCalledWith(
                expect.objectContaining({
                    props: MAIL_TEMPLATE_PREVIEW_REGISTRY['security-alert'].demoProps,
                }),
            );
            expect(JSON.stringify(renderService.renderHtml.mock.calls)).not.toContain(
                'recipient@example.com',
            );
            expect(auditRepository.create).toHaveBeenCalledWith({
                action: 'mail_test_email_requested',
                adminUserId: 'admin-1',
                payload: {
                    outboxId: 'outbox-1',
                    recipientMatchesAdmin: false,
                },
            });
        });

        it('should audit when requested recipient matches admin email case-insensitively', async () => {
            mailService.createDeliveryRequest.mockResolvedValue({ outboxId: 'outbox-1' });

            await service.sendTestEmail(
                ' Admin@Example.com ',
                'security-alert',
                'admin-1',
                'admin@example.com',
            );

            expect(auditRepository.create).toHaveBeenCalledWith({
                action: 'mail_test_email_requested',
                adminUserId: 'admin-1',
                payload: {
                    outboxId: 'outbox-1',
                    recipientMatchesAdmin: true,
                },
            });
        });

        it('should throw an error when templateId does not exist in registry', async () => {
            await expect(
                service.sendTestEmail(
                    'recipient@example.com',
                    'invalid-template',
                    'admin-1',
                    'admin@example.com',
                ),
            ).rejects.toThrow('Mail template with ID "invalid-template" not found');
        });
    });
});
