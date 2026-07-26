import { Test, TestingModule } from '@nestjs/testing';
import { ConfigService } from '@nestjs/config';
import { NotFoundException, ServiceUnavailableException } from '@nestjs/common';
import { createHash, createHmac } from 'crypto';
import { AccountSettingsService } from '@/module-account/services/account-settings.service';
import { UserRepository } from '@/module-user/repository/user.repository';
import { UserPasswordService } from '@/module-user/services/user-password.service';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { MailReadinessService } from '@/module-mail/services/mail-readiness.service';
import { MailSettingsService } from '@/module-mail/services/mail-settings.service';
import { PasswordResetChallengeRepository } from '@/module-account/repository/password-reset-challenge.repository';
import { EmailChangeChallengeRepository } from '@/module-account/repository/email-change-challenge.repository';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { MAIL_SERVICE } from '@/module-mail/constants/mail.constants';
import { AccountSecurityRepository } from '@/module-account/repository/account-security.repository';
import {
    RepositoryTransaction,
    TransactionRepository,
} from '@/module-drizzle/repository/transaction.repository';
import { SessionService } from '@/module-auth-token/services/session.service';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { IdentityIntegrationService } from '@/module-integration/services/identity-integration.service';

describe('AccountSettingsService', () => {
    let service: AccountSettingsService;
    let transactionRepository: {
        run: jest.Mock;
    };
    let accountSecurityRepository: {
        findActiveSessions: jest.Mock;
        findSessionByIdForUser: jest.Mock;
        findActiveOtherSessionIds: jest.Mock;
        findActiveSessionIds: jest.Mock;
        updatePassword: jest.Mock;
        updateEmail: jest.Mock;
        deactivateUser: jest.Mock;
    };
    const transaction = {} as RepositoryTransaction;
    let userRepository: {
        findById: jest.Mock;
        findByNameOrEmail: jest.Mock;
        findByEmail: jest.Mock;
        update: jest.Mock;
    };
    let userPasswordService: {
        comparePassword: jest.Mock;
        hashPassword: jest.Mock;
    };
    let redisTokenService: {
        set: jest.Mock;
        get: jest.Mock;
        del: jest.Mock;
        revokeSession: jest.Mock;
    };
    let mailReadinessService: {
        assertMailReadyForCriticalFlow: jest.Mock;
    };
    let mailSettingsRepository: {
        findSingleton: jest.Mock;
    };
    let mailSettingsService: {
        getCachedSettings: jest.Mock;
    };
    let passwordResetChallengeRepository: {
        create: jest.Mock;
        findBySelector: jest.Mock;
        findActiveBySelector: jest.Mock;
        invalidateUserChallenges: jest.Mock;
        update: jest.Mock;
        consume: jest.Mock;
        incrementAttempts: jest.Mock;
    };
    let emailChangeChallengeRepository: {
        create: jest.Mock;
        findActiveByUserId: jest.Mock;
        invalidateUserChallenges: jest.Mock;
        update: jest.Mock;
        consume: jest.Mock;
        incrementAttempts: jest.Mock;
    };
    let mailTemplateService: {
        sendPasswordReset: jest.Mock;
        sendVerificationCode: jest.Mock;
        sendEmailChangeConfirmation: jest.Mock;
        sendSecurityAlert: jest.Mock;
    };
    let sessionService: {
        revokeSpecificSession: jest.Mock;
        revokeOtherSessions: jest.Mock;
        revokeUserSessions: jest.Mock;
    };
    let mailService: {
        createDeliveryRequest: jest.Mock;
    };
    let configService: {
        getOrThrow: jest.Mock;
        get: jest.Mock;
    };
    let reauthConfirmationService: {
        consumeReauthConfirmation: jest.Mock;
    };
    let securityEventService: {
        recordPasswordChanged: jest.Mock;
        recordEmailChangeRequested: jest.Mock;
    };
    let identityIntegrationService: {
        recordUserDisabled: jest.Mock;
        recordUserDeleted: jest.Mock;
    };

    beforeEach(async () => {
        transactionRepository = {
            run: jest
                .fn()
                .mockImplementation((work: (tx: RepositoryTransaction) => unknown) =>
                    work(transaction),
                ),
        };
        accountSecurityRepository = {
            findActiveSessions: jest.fn(),
            findSessionByIdForUser: jest.fn(),
            findActiveOtherSessionIds: jest.fn(),
            findActiveSessionIds: jest.fn().mockResolvedValue([]),
            updatePassword: jest.fn(),
            updateEmail: jest.fn(),
            deactivateUser: jest.fn(),
        };
        userRepository = {
            findById: jest.fn(),
            findByNameOrEmail: jest.fn(),
            findByEmail: jest.fn(),
            update: jest.fn(),
        };
        userPasswordService = {
            comparePassword: jest.fn(),
            hashPassword: jest.fn(),
        };
        redisTokenService = {
            set: jest.fn(),
            get: jest.fn(),
            del: jest.fn(),
            revokeSession: jest.fn(),
        };
        mailReadinessService = {
            assertMailReadyForCriticalFlow: jest.fn().mockResolvedValue(undefined),
        };
        mailSettingsRepository = {
            findSingleton: jest.fn(),
        };
        mailSettingsService = {
            getCachedSettings: jest
                .fn()
                .mockImplementation(
                    (tx?: unknown) => mailSettingsRepository.findSingleton(tx) as Promise<unknown>,
                ),
        };
        passwordResetChallengeRepository = {
            create: jest.fn(),
            findBySelector: jest.fn(),
            findActiveBySelector: jest.fn(),
            invalidateUserChallenges: jest.fn(),
            update: jest.fn(),
            consume: jest.fn(),
            incrementAttempts: jest.fn(),
        };
        emailChangeChallengeRepository = {
            create: jest.fn(),
            findActiveByUserId: jest.fn(),
            invalidateUserChallenges: jest.fn(),
            update: jest.fn(),
            consume: jest.fn(),
            incrementAttempts: jest.fn(),
        };
        mailTemplateService = {
            sendPasswordReset: jest.fn(),
            sendVerificationCode: jest.fn(),
            sendEmailChangeConfirmation: jest.fn(),
            sendSecurityAlert: jest.fn(),
        };
        sessionService = {
            revokeSpecificSession: jest.fn().mockResolvedValue(undefined),
            revokeOtherSessions: jest.fn().mockResolvedValue(1),
            revokeUserSessions: jest.fn().mockResolvedValue(1),
        };
        mailService = {
            createDeliveryRequest: jest.fn(),
        };
        configService = {
            getOrThrow: jest.fn().mockReturnValue('30m'), // JWT_ACCESS_TOKEN_TTL
            get: jest.fn().mockImplementation((key: string) => {
                if (key === 'REAUTH_ENFORCEMENT_ENABLED') {
                    return false; // Disabled by default in existing tests
                }
                return 'default-pepper-for-tests';
            }),
        };
        reauthConfirmationService = {
            consumeReauthConfirmation: jest.fn(),
        };
        securityEventService = {
            recordPasswordChanged: jest.fn(),
            recordEmailChangeRequested: jest.fn(),
        };
        identityIntegrationService = {
            recordUserDisabled: jest.fn().mockResolvedValue(undefined),
            recordUserDeleted: jest.fn().mockResolvedValue(undefined),
        };

        const module: TestingModule = await Test.createTestingModule({
            providers: [
                AccountSettingsService,
                { provide: TransactionRepository, useValue: transactionRepository },
                { provide: AccountSecurityRepository, useValue: accountSecurityRepository },
                { provide: UserRepository, useValue: userRepository },
                { provide: UserPasswordService, useValue: userPasswordService },
                { provide: RedisTokenService, useValue: redisTokenService },
                { provide: MailReadinessService, useValue: mailReadinessService },
                { provide: MailSettingsService, useValue: mailSettingsService },
                {
                    provide: PasswordResetChallengeRepository,
                    useValue: passwordResetChallengeRepository,
                },
                {
                    provide: EmailChangeChallengeRepository,
                    useValue: emailChangeChallengeRepository,
                },
                { provide: MailTemplateService, useValue: mailTemplateService },
                { provide: SessionService, useValue: sessionService },
                { provide: MAIL_SERVICE, useValue: mailService },
                { provide: ConfigService, useValue: configService },
                { provide: ReauthConfirmationService, useValue: reauthConfirmationService },
                { provide: SecurityEventService, useValue: securityEventService },
                { provide: IdentityIntegrationService, useValue: identityIntegrationService },
            ],
        }).compile();

        service = module.get<AccountSettingsService>(AccountSettingsService);
    });

    describe('getActiveSessions', () => {
        it('should return active sessions from db', async () => {
            const mockSessions = [
                { id: 'sess-1', ipAddress: '1.1.1.1', userAgent: 'chrome' },
                { id: 'sess-2', ipAddress: '2.2.2.2', userAgent: 'safari' },
            ];
            accountSecurityRepository.findActiveSessions.mockResolvedValue(mockSessions);

            const result = await service.getActiveSessions('user-1', 'customer', 'sess-1');
            expect(result).toEqual([
                { ...mockSessions[0], currentSession: true },
                { ...mockSessions[1], currentSession: false },
            ]);
            expect(accountSecurityRepository.findActiveSessions).toHaveBeenCalledWith(
                'user-1',
                'customer',
            );
        });
    });

    describe('revokeSession', () => {
        it('should delegate specific session revoke to SessionService', async () => {
            await service.revokeSession('user-1', 'customer', 'sess-1');

            expect(sessionService.revokeSpecificSession).toHaveBeenCalledWith({
                userId: 'user-1',
                realm: 'customer',
                sessionId: 'sess-1',
            });
            expect(accountSecurityRepository.findSessionByIdForUser).not.toHaveBeenCalled();
            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
        });

        it('should propagate SessionService ownership/not-found errors without local mutation', async () => {
            const error = new NotFoundException('Session not found');
            sessionService.revokeSpecificSession.mockRejectedValue(error);

            await expect(service.revokeSession('user-1', 'admin', 'sess-1')).rejects.toBe(error);

            expect(sessionService.revokeSpecificSession).toHaveBeenCalledWith({
                userId: 'user-1',
                realm: 'admin',
                sessionId: 'sess-1',
            });
            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
        });

        it('should not open a legacy account transaction for specific session revoke', async () => {
            await service.revokeSession('user-1', 'customer', 'sess-1');

            expect(transactionRepository.run).not.toHaveBeenCalled();
        });
    });

    describe('revokeOtherSessions', () => {
        it('should delegate revoke-other orchestration to SessionService', async () => {
            await service.revokeOtherSessions('user-1', 'customer', 'current-session');

            expect(sessionService.revokeOtherSessions).toHaveBeenCalledWith({
                userId: 'user-1',
                realm: 'customer',
                currentSessionId: 'current-session',
            });
            expect(accountSecurityRepository.findActiveOtherSessionIds).not.toHaveBeenCalled();
            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
        });

        it('should propagate SessionService revoke-other errors without local mutation', async () => {
            const error = new Error('redis unavailable');
            sessionService.revokeOtherSessions.mockRejectedValue(error);

            await expect(
                service.revokeOtherSessions('user-1', 'customer', 'current-session'),
            ).rejects.toBe(error);

            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
        });
    });

    describe('resetPasswordRequest', () => {
        it('should throw ServiceUnavailable if mail system is not ready', async () => {
            mailReadinessService.assertMailReadyForCriticalFlow.mockRejectedValue(
                new ServiceUnavailableException('mail_service_not_ready'),
            );

            await expect(service.resetPasswordRequest('test@test.com')).rejects.toThrow(
                ServiceUnavailableException,
            );
            expect(userRepository.findByNameOrEmail).not.toHaveBeenCalled();
        });

        it('should return quietly if user is not found', async () => {
            userRepository.findByNameOrEmail.mockResolvedValue([]);

            await service.resetPasswordRequest('unknown@example.com');
            expect(passwordResetChallengeRepository.create).not.toHaveBeenCalled();
        });

        it('should create DB challenge and send email if user is found', async () => {
            userRepository.findByNameOrEmail.mockResolvedValue([
                { id: 'user-1', email: 'john@example.com' },
            ]);
            mailSettingsRepository.findSingleton.mockResolvedValue({
                clientPublicUrl: 'https://app.com',
            });
            await service.resetPasswordRequest('john@example.com');

            expect(passwordResetChallengeRepository.invalidateUserChallenges).toHaveBeenCalledWith(
                'user-1',
                transaction,
            );
            expect(passwordResetChallengeRepository.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    userId: 'user-1',
                    selector: expect.any(String) as unknown as string,
                    verifierDigest: expect.any(String) as unknown as string,
                }),
                transaction,
            );
            expect(mailTemplateService.sendPasswordReset).toHaveBeenCalledWith(
                'john@example.com',
                'User',
                expect.stringContaining('https://app.com/auth/reset-password?token='),
                15,
                transaction,
            );
        });

        it('should fail and rollback transaction if email delivery (outbox) fails', async () => {
            userRepository.findByNameOrEmail.mockResolvedValue([
                { id: 'user-1', email: 'john@example.com' },
            ]);
            mailSettingsRepository.findSingleton.mockResolvedValue({
                clientPublicUrl: 'https://app.com',
            });
            mailTemplateService.sendPasswordReset.mockRejectedValue(new Error('Outbox error'));

            await expect(service.resetPasswordRequest('john@example.com')).rejects.toThrow(
                'Outbox error',
            );
        });
    });

    describe('resetPasswordConfirm', () => {
        it('should throw BadRequestException if token format is invalid', async () => {
            await expect(
                service.resetPasswordConfirm('invalid-token', {
                    token: 'invalid-token',
                    newPassword: 'new',
                }),
            ).rejects.toThrow('Invalid reset token format');
        });

        it('should throw BadRequestException if challenge not found', async () => {
            passwordResetChallengeRepository.findActiveBySelector.mockResolvedValue(null);
            await expect(
                service.resetPasswordConfirm('sel.ver', { token: 'sel.ver', newPassword: 'new' }),
            ).rejects.toThrow('Invalid or expired reset token');
        });

        it('should throw BadRequest if consume fails (race condition)', async () => {
            const verifier = 'ver';
            const verifierDigest = createHash('sha256').update(verifier).digest('hex');

            passwordResetChallengeRepository.findActiveBySelector.mockResolvedValue({
                id: 'ch-1',
                userId: 'user-1',
                verifierDigest,
                attemptCount: 0,
            });
            passwordResetChallengeRepository.consume.mockResolvedValue(false);
            userRepository.findById.mockResolvedValue({ id: 'user-1' });
            await expect(
                service.resetPasswordConfirm('sel.ver', { token: 'sel.ver', newPassword: 'new' }),
            ).rejects.toThrow('Challenge already used or expired');
            expect(sessionService.revokeUserSessions).not.toHaveBeenCalled();
        });

        it('should increment attempt count on verifier mismatch', async () => {
            passwordResetChallengeRepository.findActiveBySelector.mockResolvedValue({
                id: 'ch-1',
                verifierDigest: 'wrong-digest',
                attemptCount: 0,
            });
            await expect(
                service.resetPasswordConfirm('sel.ver', { token: 'sel.ver', newPassword: 'new' }),
            ).rejects.toThrow('Invalid or expired reset token');
            expect(passwordResetChallengeRepository.incrementAttempts).toHaveBeenCalledWith('ch-1');
            expect(sessionService.revokeUserSessions).not.toHaveBeenCalled();
        });

        it('should update password and revoke user sessions without deleting session rows', async () => {
            const verifier = 'ver';
            const verifierDigest = createHash('sha256').update(verifier).digest('hex');

            passwordResetChallengeRepository.findActiveBySelector.mockResolvedValue({
                id: 'ch-1',
                userId: 'user-1',
                verifierDigest,
                attemptCount: 0,
            });
            passwordResetChallengeRepository.consume.mockResolvedValue(true);
            userRepository.findById.mockResolvedValue({ id: 'user-1', role: UserRoles.USER });
            userPasswordService.hashPassword.mockResolvedValue('new-password-hash');

            await service.resetPasswordConfirm('sel.ver', {
                token: 'sel.ver',
                newPassword: 'new',
            });

            expect(accountSecurityRepository.updatePassword).toHaveBeenCalledWith(
                'user-1',
                'new-password-hash',
                transaction,
            );
            expect(sessionService.revokeUserSessions).toHaveBeenCalledWith(
                {
                    userId: 'user-1',
                    eventType: SecurityEventType.PASSWORD_RESET_COMPLETED,
                    fallbackRealm: 'customer',
                },
                transaction,
            );
            expect(sessionService.revokeUserSessions.mock.invocationCallOrder[0]).toBeLessThan(
                accountSecurityRepository.updatePassword.mock.invocationCallOrder[0],
            );
        });

        it('should lock challenge if max attempts reached on verifier mismatch', async () => {
            const verifierDigest = 'wrong-digest';

            passwordResetChallengeRepository.findActiveBySelector.mockResolvedValue({
                id: 'ch-1',
                userId: 'user-1',
                verifierDigest,
                attemptCount: 4, // 5th attempt will lock
            });

            await expect(
                service.resetPasswordConfirm('sel.ver', { token: 'sel.ver', newPassword: 'new' }),
            ).rejects.toThrow('Invalid or expired reset token');

            expect(passwordResetChallengeRepository.incrementAttempts).toHaveBeenCalledWith('ch-1');
        });
    });

    describe('changeEmailRequest', () => {
        it('should create DB challenge and send email if request is valid', async () => {
            userRepository.findById.mockResolvedValue({ id: 'user-1' });
            userRepository.findByEmail.mockResolvedValue(null);
            await service.changeEmailRequest(
                'user-1',
                'customer',
                'sess-1',
                'new@example.com',
                'reauth-token',
            );

            expect(emailChangeChallengeRepository.invalidateUserChallenges).toHaveBeenCalledWith(
                'user-1',
                transaction,
            );
            expect(emailChangeChallengeRepository.create).toHaveBeenCalledWith(
                expect.objectContaining({
                    userId: 'user-1',
                    newEmail: 'new@example.com',
                    codeDigest: expect.any(String) as unknown as string,
                }),
                transaction,
            );
            expect(mailTemplateService.sendEmailChangeConfirmation).toHaveBeenCalledWith(
                'new@example.com',
                'User',
                'new@example.com',
                expect.any(String) as unknown as string,
                15,
                transaction,
            );
            expect(securityEventService.recordEmailChangeRequested).toHaveBeenCalledWith(
                {
                    userId: 'user-1',
                    realm: 'customer',
                    sessionId: 'sess-1',
                    ipAddress: undefined,
                    userAgent: undefined,
                    metadata: {
                        newEmailHash: expect.any(String) as unknown as string,
                    },
                },
                transaction,
            );
        });

        it('should fail and rollback transaction if email delivery (outbox) fails', async () => {
            userRepository.findById.mockResolvedValue({ id: 'user-1' });
            userRepository.findByEmail.mockResolvedValue(null);
            mailTemplateService.sendEmailChangeConfirmation.mockRejectedValue(
                new Error('Outbox error'),
            );
            await expect(
                service.changeEmailRequest('user-1', 'customer', 'sess-1', 'new@example.com'),
            ).rejects.toThrow('Outbox error');
        });

        it('should enforce reauth when REAUTH_ENFORCEMENT_ENABLED is true', async () => {
            configService.get.mockImplementation((key: string) => {
                if (key === 'REAUTH_ENFORCEMENT_ENABLED') {
                    return true;
                }
                return 'default-pepper-for-tests';
            });
            userRepository.findById.mockResolvedValue({ id: 'user-1' });
            userRepository.findByEmail.mockResolvedValue(null);

            // Reauth fails
            reauthConfirmationService.consumeReauthConfirmation.mockResolvedValue(false);

            await expect(
                service.changeEmailRequest(
                    'user-1',
                    'customer',
                    'sess-1',
                    'new@example.com',
                    'invalid-token',
                ),
            ).rejects.toThrow('Invalid or expired re-authentication confirmation token');

            // Token missing
            await expect(
                service.changeEmailRequest('user-1', 'customer', 'sess-1', 'new@example.com'),
            ).rejects.toThrow('Re-authentication confirmation token is required');
        });
    });

    describe('changeEmailConfirm', () => {
        it('should throw BadRequest if no active challenge found', async () => {
            emailChangeChallengeRepository.findActiveByUserId.mockResolvedValue(null);
            await expect(service.changeEmailConfirm('user-1', '123456')).rejects.toThrow(
                'No pending email change request found',
            );
        });

        it('should increment attempt count on verification code mismatch', async () => {
            emailChangeChallengeRepository.findActiveByUserId.mockResolvedValue({
                id: 'ch-1',
                userId: 'user-1',
                codeDigest: 'wrong-digest',
                attemptCount: 0,
            });

            await expect(service.changeEmailConfirm('user-1', '123456')).rejects.toThrow(
                'Invalid or expired verification code',
            );

            expect(emailChangeChallengeRepository.incrementAttempts).toHaveBeenCalledWith('ch-1');
        });

        it('should lock challenge if max attempts reached on verification code mismatch', async () => {
            emailChangeChallengeRepository.findActiveByUserId.mockResolvedValue({
                id: 'ch-1',
                userId: 'user-1',
                codeDigest: 'wrong-digest',
                attemptCount: 4,
            });

            await expect(service.changeEmailConfirm('user-1', '123456')).rejects.toThrow(
                'Invalid or expired verification code',
            );

            expect(emailChangeChallengeRepository.incrementAttempts).toHaveBeenCalledWith('ch-1');
        });

        it('should throw BadRequest if consume fails (race condition)', async () => {
            const code = '123456';
            const codeDigest = createHmac('sha256', 'default-pepper-for-tests')
                .update(code)
                .digest('hex');

            emailChangeChallengeRepository.findActiveByUserId.mockResolvedValue({
                id: 'ch-1',
                newEmail: 'new@test.com',
                codeDigest,
            });
            emailChangeChallengeRepository.consume.mockResolvedValue(false);
            userRepository.findById.mockResolvedValue({ id: 'user-1' });
            await expect(service.changeEmailConfirm('user-1', code)).rejects.toThrow(
                'Challenge already used or expired',
            );
        });

        it('should update email and consume challenge if code matches', async () => {
            const code = '123456';
            const codeDigest = createHmac('sha256', 'default-pepper-for-tests')
                .update(code)
                .digest('hex');

            emailChangeChallengeRepository.findActiveByUserId.mockResolvedValue({
                id: 'ch-1',
                newEmail: 'new@test.com',
                codeDigest,
            });
            emailChangeChallengeRepository.consume.mockResolvedValue(true);
            userRepository.findById.mockResolvedValue({ id: 'user-1' });
            await service.changeEmailConfirm('user-1', code);

            expect(emailChangeChallengeRepository.consume).toHaveBeenCalledWith(
                'ch-1',
                transaction,
            );
            expect(accountSecurityRepository.updateEmail).toHaveBeenCalledWith(
                'user-1',
                'new@test.com',
                transaction,
            );
            expect(sessionService.revokeUserSessions).toHaveBeenCalledWith(
                { userId: 'user-1' },
                transaction,
            );
        });
    });

    describe('changePassword', () => {
        it('should update password when valid and bypass re-auth when disabled, recording a security event', async () => {
            userRepository.findById.mockResolvedValue({ id: 'user-1', password: 'old-hash' });
            userPasswordService.comparePassword.mockResolvedValue(true);
            userPasswordService.hashPassword.mockResolvedValue('new-hash');

            const dto = { oldPassword: 'old', newPassword: 'new' };
            await service.changePassword('user-1', 'customer', 'sess-1', dto, undefined, {
                ipAddress: '127.0.0.1',
                userAgent: 'test-agent',
            });

            expect(userRepository.update).toHaveBeenCalledWith(
                'user-1',
                { password: 'new-hash' },
                transaction,
            );
            expect(sessionService.revokeUserSessions).toHaveBeenCalledWith(
                { userId: 'user-1' },
                transaction,
            );
            expect(securityEventService.recordPasswordChanged).toHaveBeenCalledWith(
                {
                    userId: 'user-1',
                    realm: 'customer',
                    sessionId: 'sess-1',
                    ipAddress: '127.0.0.1',
                    userAgent: 'test-agent',
                    metadata: {
                        revokedSessionCount: 1,
                    },
                },
                transaction,
            );
        });

        it('should enforce re-auth when REAUTH_ENFORCEMENT_ENABLED is true', async () => {
            configService.get.mockImplementation((key: string) => {
                if (key === 'REAUTH_ENFORCEMENT_ENABLED') {
                    return true;
                }
                return 'default-pepper-for-tests';
            });
            userRepository.findById.mockResolvedValue({ id: 'user-1', password: 'old-hash' });
            userPasswordService.comparePassword.mockResolvedValue(true);
            userPasswordService.hashPassword.mockResolvedValue('new-hash');

            const dto = { oldPassword: 'old', newPassword: 'new' };

            // Reauth fails
            reauthConfirmationService.consumeReauthConfirmation.mockResolvedValue(false);
            await expect(
                service.changePassword('user-1', 'customer', 'sess-1', dto, 'invalid-token'),
            ).rejects.toThrow('Invalid or expired re-authentication confirmation token');

            // Token missing
            await expect(
                service.changePassword('user-1', 'customer', 'sess-1', dto),
            ).rejects.toThrow('Re-authentication confirmation token is required');
        });
    });
});
