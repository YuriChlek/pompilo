import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { ReauthConfirmationRepository } from '@/module-auth-token/repository/reauth-confirmation.repository';
import { UserRepository } from '@/module-user/repository/user.repository';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { buildUserEntity } from '../../fixtures/users.fixtures';
import { ReauthConfirmationSelect } from '@/module-auth-token/schemas/reauth-confirmations.schema';
import { UnauthorizedException } from '@nestjs/common';
import { createHash } from 'crypto';

describe('ReauthConfirmationService', () => {
    let service: ReauthConfirmationService;
    let mockReauthConfirmationRepository: jest.Mocked<ReauthConfirmationRepository>;
    let mockUserRepository: jest.Mocked<UserRepository>;
    let mockSecurityEventService: jest.Mocked<SecurityEventService>;

    const now = new Date('2026-06-23T12:00:00Z');

    const buildMockConfirmation = (
        overrides: Partial<ReauthConfirmationSelect> = {},
    ): ReauthConfirmationSelect => ({
        id: overrides.id ?? 'confirmation-uuid',
        userId: overrides.userId ?? 'user-uuid',
        realm: overrides.realm ?? 'customer',
        sessionId: overrides.sessionId ?? 'session-uuid',
        actionScope: overrides.actionScope ?? 'email_change',
        confirmationTokenHash: overrides.confirmationTokenHash ?? 'token-hash',
        expiresAt: overrides.expiresAt ?? new Date(now.getTime() + 300000),
        consumedAt: overrides.consumedAt ?? null,
        createdAt: overrides.createdAt ?? now,
    });

    beforeEach(() => {
        mockReauthConfirmationRepository = {
            save: jest.fn(),
            findById: jest.fn(),
            findByTokenHash: jest.fn(),
            consume: jest.fn(),
            consumeByTokenHash: jest.fn(),
            deleteExpired: jest.fn(),
        } as unknown as jest.Mocked<ReauthConfirmationRepository>;

        mockUserRepository = {
            findById: jest.fn(),
        } as unknown as jest.Mocked<UserRepository>;

        mockSecurityEventService = {
            recordReauthPassed: jest.fn(),
            recordReauthFailed: jest.fn(),
        } as unknown as jest.Mocked<SecurityEventService>;

        service = new ReauthConfirmationService(
            mockReauthConfirmationRepository,
            mockUserRepository,
            mockSecurityEventService,
        );

        jest.clearAllMocks();
        jest.restoreAllMocks();
    });

    describe('createReauthConfirmation', () => {
        it('should throw UnauthorizedException if user does not exist', async () => {
            mockUserRepository.findById.mockResolvedValue(null);

            await expect(
                service.createReauthConfirmation(
                    'user-uuid',
                    'customer',
                    'session-uuid',
                    'email_change',
                    'password123',
                    { ipAddress: '127.0.0.1', userAgent: 'userAgent' },
                    now,
                ),
            ).rejects.toThrow(UnauthorizedException);

            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockSecurityEventService.recordReauthFailed).not.toHaveBeenCalled();
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockReauthConfirmationRepository.save).not.toHaveBeenCalled();
        });

        it('should record failure event and throw if password is invalid', async () => {
            const user = buildUserEntity({ id: 'user-uuid', password: 'hashed-password' });

            mockUserRepository.findById.mockResolvedValue(user);
            jest.spyOn(Argon2HashUtil, 'compare').mockResolvedValue(false);

            await expect(
                service.createReauthConfirmation(
                    'user-uuid',
                    'customer',
                    'session-uuid',
                    'email_change',
                    'wrong-password',
                    { ipAddress: '127.0.0.1', userAgent: 'userAgent' },
                    now,
                ),
            ).rejects.toThrow(UnauthorizedException);

            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockSecurityEventService.recordReauthFailed).toHaveBeenCalledWith({
                userId: 'user-uuid',
                sessionId: 'session-uuid',
                realm: 'customer',
                ipAddress: '127.0.0.1',
                userAgent: 'userAgent',
                metadata: {
                    actionScope: 'email_change',
                },
            });
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockReauthConfirmationRepository.save).not.toHaveBeenCalled();
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockSecurityEventService.recordReauthPassed).not.toHaveBeenCalled();
        });

        it('should successfully create reauth confirmation, return raw token, store hash, and log success event', async () => {
            const user = buildUserEntity({ id: 'user-uuid', password: 'hashed-password' });

            mockUserRepository.findById.mockResolvedValue(user);
            jest.spyOn(Argon2HashUtil, 'compare').mockResolvedValue(true);

            mockReauthConfirmationRepository.save.mockResolvedValue(buildMockConfirmation());

            const rawToken = await service.createReauthConfirmation(
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                'correct-password',
                { ipAddress: '127.0.0.1', userAgent: 'userAgent' },
                now,
            );

            expect(rawToken).toBeDefined();
            expect(typeof rawToken).toBe('string');
            expect(rawToken.length).toBe(64); // 32 bytes in hex = 64 chars

            const expectedHash = createHash('sha256').update(rawToken).digest('hex');

            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockReauthConfirmationRepository.save).toHaveBeenCalledWith({
                userId: 'user-uuid',
                realm: 'customer',
                sessionId: 'session-uuid',
                actionScope: 'email_change',
                confirmationTokenHash: expectedHash,
                expiresAt: new Date(now.getTime() + 5 * 60 * 1000),
                createdAt: now,
            });

            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockSecurityEventService.recordReauthPassed).toHaveBeenCalledWith({
                userId: 'user-uuid',
                sessionId: 'session-uuid',
                realm: 'customer',
                ipAddress: '127.0.0.1',
                userAgent: 'userAgent',
                metadata: {
                    actionScope: 'email_change',
                },
            });
        });
    });

    describe('validateReauthConfirmationToken', () => {
        it('should return false if token is not found', async () => {
            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(null);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
        });

        it('should return false if userId does not match', async () => {
            const confirmation = buildMockConfirmation({ userId: 'other-user' });

            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(confirmation);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
        });

        it('should return false if realm does not match', async () => {
            const confirmation = buildMockConfirmation({ realm: 'admin' });

            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(confirmation);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
        });

        it('should return false if sessionId does not match', async () => {
            const confirmation = buildMockConfirmation({ sessionId: 'other-session' });

            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(confirmation);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
        });

        it('should return false if actionScope does not match', async () => {
            const confirmation = buildMockConfirmation({ actionScope: 'other-scope' });

            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(confirmation);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
        });

        it('should return false if already consumed', async () => {
            const confirmation = buildMockConfirmation({ consumedAt: new Date() });

            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(confirmation);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
        });

        it('should return false if expired', async () => {
            const confirmation = buildMockConfirmation({
                expiresAt: new Date(now.getTime() - 1000),
            });

            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(confirmation);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
        });

        it('should return true if token matches and is valid', async () => {
            const confirmation = buildMockConfirmation();

            mockReauthConfirmationRepository.findByTokenHash.mockResolvedValue(confirmation);

            const result = await service.validateReauthConfirmationToken(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(true);
        });
    });

    describe('consumeReauthConfirmation', () => {
        it('should return false if repository returns false (not found, mismatched, or already consumed/expired)', async () => {
            mockReauthConfirmationRepository.consumeByTokenHash.mockResolvedValue(false);

            const result = await service.consumeReauthConfirmation(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(false);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockReauthConfirmationRepository.consumeByTokenHash).toHaveBeenCalledWith(
                createHash('sha256').update('raw-token').digest('hex'),
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
                undefined,
            );
        });

        it('should return true if repository successfully consumes (updates) the token', async () => {
            mockReauthConfirmationRepository.consumeByTokenHash.mockResolvedValue(true);

            const result = await service.consumeReauthConfirmation(
                'raw-token',
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
            );

            expect(result).toBe(true);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockReauthConfirmationRepository.consumeByTokenHash).toHaveBeenCalledWith(
                createHash('sha256').update('raw-token').digest('hex'),
                'user-uuid',
                'customer',
                'session-uuid',
                'email_change',
                now,
                undefined,
            );
        });
    });

    describe('expireReauthConfirmations', () => {
        it('should delete expired records and return the count', async () => {
            mockReauthConfirmationRepository.deleteExpired.mockResolvedValue(2);

            const count = await service.expireReauthConfirmations(0, now);

            expect(count).toBe(2);
            // eslint-disable-next-line @typescript-eslint/unbound-method
            expect(mockReauthConfirmationRepository.deleteExpired).toHaveBeenCalledWith(0, now);
        });
    });
});
