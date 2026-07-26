/* eslint-disable @typescript-eslint/unbound-method, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call, @typescript-eslint/no-unsafe-return */
import { EmailVerificationService } from '@/module-auth-token/services/email-verification.service';
import { EmailVerificationRepository } from '@/module-auth-token/repository/email-verification.repository';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { UserRepository } from '@/module-user/repository/user.repository';
import { buildUserEntity } from '../../fixtures/users.fixtures';
import { EmailVerificationSelect } from '@/module-auth-token/schemas/email-verifications.schema';
import { BadRequestException, ConflictException } from '@nestjs/common';
import { createHash } from 'crypto';
import {
    RepositoryTransaction,
    TransactionRepository,
} from '@/module-drizzle/repository/transaction.repository';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';

describe('EmailVerificationService', () => {
    let service: EmailVerificationService;
    let mockEmailVerificationRepository: jest.Mocked<EmailVerificationRepository>;
    let mockMailTemplateService: jest.Mocked<MailTemplateService>;
    let mockUserRepository: jest.Mocked<UserRepository>;
    let mockTransactionRepository: jest.Mocked<TransactionRepository>;
    let mockSecurityEventService: jest.Mocked<SecurityEventService>;
    const transaction = {} as RepositoryTransaction;

    const buildMockVerification = (
        overrides: Partial<EmailVerificationSelect> = {},
    ): EmailVerificationSelect => ({
        id: overrides.id ?? 'verification-uuid',
        userId: overrides.userId ?? 'user-uuid',
        tokenHash: overrides.tokenHash ?? 'token-hash',
        expiresAt: overrides.expiresAt ?? new Date(Date.now() + 30 * 60 * 1000),
        consumedAt: overrides.consumedAt ?? null,
        createdAt: overrides.createdAt ?? new Date(),
    });

    beforeEach(() => {
        mockEmailVerificationRepository = {
            save: jest.fn(),
            findByTokenHash: jest.fn(),
            findActiveByUserId: jest.fn(),
            consume: jest.fn(),
            deleteExpired: jest.fn(),
        } as unknown as jest.Mocked<EmailVerificationRepository>;

        mockMailTemplateService = {
            sendEmailVerificationOrThrow: jest.fn(),
        } as unknown as jest.Mocked<MailTemplateService>;

        mockUserRepository = {
            findById: jest.fn(),
            update: jest.fn(),
        } as unknown as jest.Mocked<UserRepository>;

        mockTransactionRepository = {
            run: jest.fn(work => work(transaction)),
        } as unknown as jest.Mocked<TransactionRepository>;

        mockSecurityEventService = {
            recordEmailVerified: jest.fn(),
        } as unknown as jest.Mocked<SecurityEventService>;

        mockEmailVerificationRepository.consume.mockResolvedValue(true);

        service = new EmailVerificationService(
            mockEmailVerificationRepository,
            mockMailTemplateService,
            mockUserRepository,
            mockTransactionRepository,
            mockSecurityEventService,
        );

        jest.clearAllMocks();
        jest.restoreAllMocks();
    });

    describe('createVerificationToken', () => {
        it('should generate a raw token, store the hash in db and return raw token', async () => {
            mockEmailVerificationRepository.save.mockResolvedValue(buildMockVerification());

            const rawToken = await service.createVerificationToken('user-uuid');

            expect(rawToken).toBeDefined();
            expect(typeof rawToken).toBe('string');
            expect(rawToken.length).toBe(64); // 32 bytes in hex = 64 characters

            const expectedHash = createHash('sha256').update(rawToken).digest('hex');

            expect(mockEmailVerificationRepository.save).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    tokenHash: expectedHash,
                    expiresAt: expect.any(Date),
                },
                undefined,
            );
        });
    });

    describe('sendVerificationEmail', () => {
        it('should send verification link constructed with CLIENT_ORIGIN', async () => {
            process.env.CLIENT_ORIGIN = 'http://test-client.local';

            await service.sendVerificationEmail(
                'user-uuid',
                'user@example.com',
                'User Name',
                'raw-token',
            );

            expect(mockMailTemplateService.sendEmailVerificationOrThrow).toHaveBeenCalledWith(
                'user@example.com',
                'User Name',
                'http://test-client.local/verify-email?token=raw-token',
                undefined,
            );
        });
    });

    describe('verifyEmail', () => {
        it('should throw BadRequestException if verification token does not exist', async () => {
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(null);

            await expect(service.verifyEmail('raw-token')).rejects.toThrow(BadRequestException);
        });

        it('should throw ConflictException if verification token was already consumed', async () => {
            const verification = buildMockVerification({ consumedAt: new Date() });
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(verification);

            await expect(service.verifyEmail('raw-token')).rejects.toThrow(ConflictException);
        });

        it('should throw BadRequestException if token is expired', async () => {
            const verification = buildMockVerification({ expiresAt: new Date(Date.now() - 1000) });
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(verification);

            await expect(service.verifyEmail('raw-token')).rejects.toThrow(BadRequestException);
        });

        it('should throw BadRequestException if user does not exist', async () => {
            const verification = buildMockVerification();
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(verification);
            mockUserRepository.findById.mockResolvedValue(null);

            await expect(service.verifyEmail('raw-token')).rejects.toThrow(BadRequestException);
        });

        it('should throw ConflictException if user email is already verified', async () => {
            const verification = buildMockVerification();
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(verification);
            const user = buildUserEntity({ id: 'user-uuid', emailVerifiedAt: new Date() });
            mockUserRepository.findById.mockResolvedValue(user);

            await expect(service.verifyEmail('raw-token')).rejects.toThrow(ConflictException);
        });

        it('should successfully verify email, update user verification status and consume token', async () => {
            const verification = buildMockVerification();
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(verification);
            const user = buildUserEntity({ id: 'user-uuid', emailVerifiedAt: null });
            mockUserRepository.findById.mockResolvedValue(user);

            const result = await service.verifyEmail('raw-token');

            expect(result).toBe(true);
            expect(mockUserRepository.update).toHaveBeenCalledWith(
                'user-uuid',
                { emailVerifiedAt: expect.any(Date) },
                transaction,
            );
            expect(mockEmailVerificationRepository.consume).toHaveBeenCalledWith(
                verification.id,
                expect.any(Date),
                transaction,
            );
            expect(mockSecurityEventService.recordEmailVerified).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    metadata: {},
                },
                transaction,
            );
        });

        it('runs the integration callback inside the verification transaction', async () => {
            const verification = buildMockVerification();
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(verification);
            const user = buildUserEntity({ id: 'user-uuid', emailVerifiedAt: null });
            mockUserRepository.findById.mockResolvedValue(user);
            const onVerified = jest.fn().mockResolvedValue(undefined);

            await service.verifyEmail('raw-token', undefined, onVerified);

            expect(onVerified).toHaveBeenCalledWith(user, transaction);
        });

        it('rejects a concurrent attempt when atomic token consumption loses the race', async () => {
            const verification = buildMockVerification();
            mockEmailVerificationRepository.findByTokenHash.mockResolvedValue(verification);
            mockUserRepository.findById.mockResolvedValue(
                buildUserEntity({ id: 'user-uuid', emailVerifiedAt: null }),
            );
            mockEmailVerificationRepository.consume.mockResolvedValue(false);

            await expect(service.verifyEmail('raw-token')).rejects.toThrow(ConflictException);
            expect(mockUserRepository.update).not.toHaveBeenCalled();
            expect(mockSecurityEventService.recordEmailVerified).not.toHaveBeenCalled();
        });
    });

    describe('resendVerification', () => {
        it('should throw BadRequestException if user does not exist', async () => {
            mockUserRepository.findById.mockResolvedValue(null);

            await expect(
                service.resendVerification('user-uuid', 'email@example.com', 'Name'),
            ).rejects.toThrow(BadRequestException);
        });

        it('should throw ConflictException if user is already verified', async () => {
            const user = buildUserEntity({ id: 'user-uuid', emailVerifiedAt: new Date() });
            mockUserRepository.findById.mockResolvedValue(user);

            await expect(
                service.resendVerification('user-uuid', 'email@example.com', 'Name'),
            ).rejects.toThrow(ConflictException);
        });

        it('should throw BadRequestException if active token was created less than 1 minute ago', async () => {
            const user = buildUserEntity({ id: 'user-uuid', emailVerifiedAt: null });
            mockUserRepository.findById.mockResolvedValue(user);

            const recentVerification = buildMockVerification({
                createdAt: new Date(Date.now() - 30 * 1000), // 30s ago
            });
            mockEmailVerificationRepository.findActiveByUserId.mockResolvedValue(
                recentVerification,
            );

            await expect(
                service.resendVerification('user-uuid', 'email@example.com', 'Name'),
            ).rejects.toThrow(BadRequestException);
        });

        it('should generate new token and send email if validation checks pass', async () => {
            const user = buildUserEntity({ id: 'user-uuid', emailVerifiedAt: null });
            mockUserRepository.findById.mockResolvedValue(user);

            mockEmailVerificationRepository.findActiveByUserId.mockResolvedValue(null);
            mockEmailVerificationRepository.save.mockResolvedValue(buildMockVerification());

            await service.resendVerification('user-uuid', 'email@example.com', 'Name');

            expect(mockEmailVerificationRepository.save).toHaveBeenCalled();
            expect(mockMailTemplateService.sendEmailVerificationOrThrow).toHaveBeenCalled();
        });
    });
});
