import { SessionService } from '@/module-auth-token/services/session.service';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { KnownDeviceRepository } from '@/module-auth-token/repository/known-device.repository';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { SessionSelect } from '@/module-auth-token/schemas/sessions.schema';
import { KnownDeviceSelect } from '@/module-auth-token/schemas/known-devices.schema';
import { ConfigService } from '@nestjs/config';
import { NotFoundException, ForbiddenException } from '@nestjs/common';
import { SecurityEventService } from '@/module-auth-token/services/security-event.service';
import {
    RepositoryTransaction,
    TransactionRepository,
} from '@/module-drizzle/repository/transaction.repository';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { ReauthConfirmationService } from '@/module-auth-token/services/reauth-confirmation.service';
import { SecurityEventType } from '@/module-auth-token/enums/security-event.enums';
import { MailTemplateService } from '@/module-mail/services/mail-template.service';
import { UserRepository } from '@/module-user/repository/user.repository';

const buildMockSession = (overrides: Partial<SessionSelect> = {}): SessionSelect => ({
    id: overrides.id ?? 'session-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    knownDeviceId: overrides.knownDeviceId ?? 'known-device-uuid',
    deviceId: overrides.deviceId ?? 'device-uuid',
    ipAddress: overrides.ipAddress ?? null,
    userAgent: overrides.userAgent ?? null,
    createdAt: overrides.createdAt ?? new Date(),
    updatedAt: overrides.updatedAt ?? new Date(),
    lastSeenAt: overrides.lastSeenAt ?? new Date(),
    expiresAt: overrides.expiresAt ?? new Date(),
    revokedAt: overrides.revokedAt ?? null,
    lastCountry: overrides.lastCountry ?? null,
    lastRegion: overrides.lastRegion ?? null,
    lastCity: overrides.lastCity ?? null,
    riskScore: overrides.riskScore ?? 0,
    riskReason: overrides.riskReason ?? null,
});

const buildMockDevice = (overrides: Partial<KnownDeviceSelect> = {}): KnownDeviceSelect => ({
    id: overrides.id ?? 'known-device-uuid',
    userId: overrides.userId ?? 'user-uuid',
    realm: overrides.realm ?? 'customer',
    deviceId: overrides.deviceId ?? 'device-uuid',
    trustedAt: null,
    trustExpiresAt: null,
    revokedAt: overrides.revokedAt ?? null,
    firstSeenAt: new Date(),
    lastSeenAt: new Date(),
    lastIpAddress: null,
    lastCountry: null,
    lastRegion: null,
    lastCity: null,
    lastUserAgent: null,
    createdAt: new Date(),
    updatedAt: new Date(),
});

describe('SessionService', () => {
    let service: SessionService;
    let sessionRepository: {
        save: jest.MockedFunction<SessionRepository['save']>;
        findById: jest.MockedFunction<SessionRepository['findById']>;
        findReusable: jest.MockedFunction<SessionRepository['findReusable']>;
        reuseSession: jest.MockedFunction<SessionRepository['reuseSession']>;
        update: jest.MockedFunction<SessionRepository['update']>;
        revoke: jest.MockedFunction<SessionRepository['revoke']>;
        revokeOther: jest.MockedFunction<SessionRepository['revokeOther']>;
        revokeAll: jest.MockedFunction<SessionRepository['revokeAll']>;
        findActiveByUserRealm: jest.MockedFunction<SessionRepository['findActiveByUserRealm']>;
        findActiveByUser: jest.MockedFunction<SessionRepository['findActiveByUser']>;
        findUnrevokedOtherByUserRealm: jest.MockedFunction<
            SessionRepository['findUnrevokedOtherByUserRealm']
        >;
        findUnrevokedByUserRealm: jest.MockedFunction<
            SessionRepository['findUnrevokedByUserRealm']
        >;
    };
    let knownDeviceRepository: {
        findById: jest.MockedFunction<KnownDeviceRepository['findById']>;
    };
    let authTokenRepository: {
        revokeTokensBySession: jest.MockedFunction<AuthTokenRepository['revokeTokensBySession']>;
    };
    let securityEventService: {
        recordLogout: jest.MockedFunction<SecurityEventService['recordLogout']>;
        recordSessionRevoked: jest.MockedFunction<SecurityEventService['recordSessionRevoked']>;
        recordRevokeOtherSessions: jest.MockedFunction<
            SecurityEventService['recordRevokeOtherSessions']
        >;
        recordRevokeAllSessions: jest.MockedFunction<
            SecurityEventService['recordRevokeAllSessions']
        >;
        recordPasswordResetCompleted: jest.MockedFunction<
            SecurityEventService['recordPasswordResetCompleted']
        >;
    };
    let transactionRepository: {
        run: jest.MockedFunction<TransactionRepository['run']>;
    };
    let redisTokenService: {
        revokeSession: jest.MockedFunction<RedisTokenService['revokeSession']>;
        revokeToken: jest.MockedFunction<RedisTokenService['revokeToken']>;
    };
    let configService: {
        getOrThrow: jest.MockedFunction<ConfigService['getOrThrow']>;
        get?: jest.Mock;
    };
    let reauthConfirmationService: jest.Mocked<ReauthConfirmationService>;
    let userRepository: {
        findById: jest.MockedFunction<UserRepository['findById']>;
    };
    let mailTemplateService: {
        sendSecurityAlert: jest.MockedFunction<MailTemplateService['sendSecurityAlert']>;
    };

    beforeEach(() => {
        sessionRepository = {
            save: jest.fn(),
            findById: jest.fn(),
            findReusable: jest.fn(),
            reuseSession: jest.fn(),
            update: jest.fn(),
            revoke: jest.fn(),
            revokeOther: jest.fn(),
            revokeAll: jest.fn(),
            findActiveByUserRealm: jest.fn(),
            findActiveByUser: jest.fn(),
            findUnrevokedOtherByUserRealm: jest.fn(),
            findUnrevokedByUserRealm: jest.fn(),
        };
        knownDeviceRepository = {
            findById: jest.fn(),
        };
        authTokenRepository = {
            revokeTokensBySession: jest.fn().mockResolvedValue(1),
        };
        securityEventService = {
            recordLogout: jest.fn().mockResolvedValue(null),
            recordSessionRevoked: jest.fn().mockResolvedValue(null),
            recordRevokeOtherSessions: jest.fn().mockResolvedValue(null),
            recordRevokeAllSessions: jest.fn().mockResolvedValue(null),
            recordPasswordResetCompleted: jest.fn().mockResolvedValue(null),
        };
        transactionRepository = {
            run: jest
                .fn()
                .mockImplementation(
                    async <T>(work: (transaction: RepositoryTransaction) => Promise<T>) =>
                        work({} as RepositoryTransaction),
                ),
        };
        redisTokenService = {
            revokeSession: jest.fn().mockResolvedValue(undefined),
            revokeToken: jest.fn().mockResolvedValue(undefined),
        };
        configService = {
            getOrThrow: jest.fn().mockImplementation((key: string) => {
                if (key === 'SESSION_MAX_TTL') return '7d';
                if (key === 'JWT_ACCESS_TOKEN_TTL') return '15m';
                if (key === 'AUTH_CLOCK_SKEW_SECONDS') return 30;
                return undefined;
            }),
            get: jest.fn().mockReturnValue(false),
        };
        reauthConfirmationService = {
            consumeReauthConfirmation: jest.fn(),
        } as unknown as jest.Mocked<ReauthConfirmationService>;

        userRepository = {
            findById: jest.fn().mockResolvedValue({
                id: 'user-uuid',
                email: 'user@email.com',
                name: 'User',
                password: 'hashed',
                role: 'user',
                isActive: true,
                createdAt: new Date(),
                updatedAt: new Date(),
                deletionScheduledAt: null,
                tokens: [],
            }),
        };
        mailTemplateService = {
            sendSecurityAlert: jest.fn().mockResolvedValue({}),
        };

        service = new SessionService(
            sessionRepository as unknown as SessionRepository,
            knownDeviceRepository as unknown as KnownDeviceRepository,
            authTokenRepository as unknown as AuthTokenRepository,
            securityEventService as unknown as SecurityEventService,
            transactionRepository as unknown as TransactionRepository,
            redisTokenService as unknown as RedisTokenService,
            configService as unknown as ConfigService,
            reauthConfirmationService,
            userRepository as unknown as UserRepository,
            mailTemplateService as unknown as MailTemplateService,
        );
    });

    describe('findReusableSession', () => {
        it('should return null if known device does not exist', async () => {
            knownDeviceRepository.findById.mockResolvedValue(null);

            const result = await service.findReusableSession(
                'user-uuid',
                'customer',
                'device-uuid',
            );

            expect(result).toBeNull();
            expect(knownDeviceRepository.findById).toHaveBeenCalledWith('device-uuid', undefined);
        });

        it('should return null if known device user/realm does not match', async () => {
            const device = buildMockDevice({ userId: 'different-user', realm: 'customer' });
            knownDeviceRepository.findById.mockResolvedValue(device);

            const result = await service.findReusableSession(
                'user-uuid',
                'customer',
                'device-uuid',
            );

            expect(result).toBeNull();
        });

        it('should return null if known device is revoked', async () => {
            const device = buildMockDevice({ revokedAt: new Date() });
            knownDeviceRepository.findById.mockResolvedValue(device);

            const result = await service.findReusableSession(
                'user-uuid',
                'customer',
                'device-uuid',
            );

            expect(result).toBeNull();
        });

        it('should call sessionRepository.findReusable and return result if device is active and matches', async () => {
            const device = buildMockDevice();
            const session = buildMockSession();
            knownDeviceRepository.findById.mockResolvedValue(device);
            sessionRepository.findReusable.mockResolvedValue(session);

            const result = await service.findReusableSession(
                'user-uuid',
                'customer',
                'known-device-uuid',
            );

            expect(result).toEqual(session);
            expect(sessionRepository.findReusable).toHaveBeenCalledWith(
                'user-uuid',
                'customer',
                'device-uuid',
                undefined,
            );
        });
    });

    describe('createSession', () => {
        it('should calculate expiresAt and save session', async () => {
            const session = buildMockSession();
            sessionRepository.save.mockResolvedValue(session);

            const metadata = { ipAddress: '1.1.1.1' };
            const result = await service.createSession(
                'user-uuid',
                'customer',
                'known-device-uuid',
                'device-uuid',
                metadata,
            );

            expect(result).toEqual(session);
            expect(sessionRepository.save).toHaveBeenCalledWith(
                expect.objectContaining({
                    userId: 'user-uuid',
                    realm: 'customer',
                    knownDeviceId: 'known-device-uuid',
                    deviceId: 'device-uuid',
                    ipAddress: '1.1.1.1',
                    expiresAt: expect.any(Date) as unknown as Date,
                }),
                undefined,
            );
        });

        it('should catch unique violation, find reusable session, and reuse it', async () => {
            const dbError = Object.assign(new Error('Unique violation'), { code: '23505' });
            sessionRepository.save.mockRejectedValue(dbError);

            const existingSession = buildMockSession({ id: 'existing-session-uuid' });
            sessionRepository.findReusable.mockResolvedValue(existingSession);

            const updatedSession = buildMockSession({
                id: 'existing-session-uuid',
                ipAddress: 'updated-ip',
            });
            sessionRepository.reuseSession.mockResolvedValue(updatedSession);

            const result = await service.createSession(
                'user-uuid',
                'customer',
                'known-device-uuid',
                'device-uuid',
                { ipAddress: 'updated-ip' },
            );

            expect(result).toEqual(updatedSession);
            expect(sessionRepository.findReusable).toHaveBeenCalledWith(
                'user-uuid',
                'customer',
                'device-uuid',
                undefined,
            );
            expect(sessionRepository.reuseSession).toHaveBeenCalledWith(
                'existing-session-uuid',
                expect.any(Date),
                { ipAddress: 'updated-ip' },
                expect.any(Date),
                undefined,
            );
        });

        it('should rethrow unique violation if no reusable session is found', async () => {
            const dbError = Object.assign(new Error('Unique violation'), { code: '23505' });
            sessionRepository.save.mockRejectedValue(dbError);
            sessionRepository.findReusable.mockResolvedValue(null);

            await expect(
                service.createSession('user-uuid', 'customer', 'known-device-uuid', 'device-uuid'),
            ).rejects.toThrow('Unique violation');
        });

        it('should rethrow other errors', async () => {
            const dbError = new Error('DB Error');
            sessionRepository.save.mockRejectedValue(dbError);

            await expect(
                service.createSession('user-uuid', 'customer', 'known-device-uuid', 'device-uuid'),
            ).rejects.toThrow('DB Error');
        });
    });

    describe('reuseSession', () => {
        it('should reuse session and return updated session', async () => {
            const session = buildMockSession();
            sessionRepository.reuseSession.mockResolvedValue(session);

            const expiresAt = new Date();
            const result = await service.reuseSession(
                'session-uuid',
                { ipAddress: '1.2.3.4' },
                expiresAt,
            );

            expect(result).toEqual(session);
            expect(sessionRepository.reuseSession).toHaveBeenCalledWith(
                'session-uuid',
                expiresAt,
                { ipAddress: '1.2.3.4' },
                expect.any(Date),
                undefined,
            );
        });

        it('should throw NotFoundException if session is not found or revoked', async () => {
            sessionRepository.reuseSession.mockResolvedValue(null);

            await expect(service.reuseSession('session-uuid')).rejects.toThrow(NotFoundException);
        });
    });

    describe('touchLastSeenAt', () => {
        it('should return null if session does not exist', async () => {
            sessionRepository.findById.mockResolvedValue(null);

            const result = await service.touchLastSeenAt('session-uuid', 60);

            expect(result).toBeNull();
        });

        it('should return null if session is revoked', async () => {
            const session = buildMockSession({ revokedAt: new Date() });
            sessionRepository.findById.mockResolvedValue(session);

            const result = await service.touchLastSeenAt('session-uuid', 60);

            expect(result).toBeNull();
        });

        it('should throttle and return session without database update if inside throttle window', async () => {
            const lastSeenAt = new Date(Date.now() - 30 * 1000); // 30 seconds ago
            const session = buildMockSession({ lastSeenAt });
            sessionRepository.findById.mockResolvedValue(session);

            const result = await service.touchLastSeenAt('session-uuid', 60); // 60s throttle window

            expect(result).toEqual(session);
            expect(sessionRepository.update).not.toHaveBeenCalled();
        });

        it('should update lastSeenAt if throttle window has expired', async () => {
            const lastSeenAt = new Date(Date.now() - 90 * 1000); // 90 seconds ago
            const session = buildMockSession({ lastSeenAt });
            const updated = buildMockSession({ lastSeenAt: new Date() });
            sessionRepository.findById.mockResolvedValue(session);
            sessionRepository.update.mockResolvedValue(updated);

            const now = new Date();
            const result = await service.touchLastSeenAt('session-uuid', 60, undefined, now); // 60s throttle window

            expect(result).toEqual(updated);
            expect(sessionRepository.update).toHaveBeenCalledWith(
                'session-uuid',
                { lastSeenAt: now },
                undefined,
            );
        });
    });

    describe('revokeSession', () => {
        it('should throw NotFoundException if session does not exist', async () => {
            sessionRepository.findById.mockResolvedValue(null);

            await expect(service.revokeSession('session-uuid')).rejects.toThrow(NotFoundException);
        });

        it('should throw ForbiddenException if userId does not match', async () => {
            const session = buildMockSession({ userId: 'owner-uuid', realm: 'customer' });
            sessionRepository.findById.mockResolvedValue(session);

            await expect(
                service.revokeSession('session-uuid', 'different-user', 'customer'),
            ).rejects.toThrow(ForbiddenException);
        });

        it('should throw ForbiddenException if realm does not match', async () => {
            const session = buildMockSession({ userId: 'owner-uuid', realm: 'customer' });
            sessionRepository.findById.mockResolvedValue(session);

            await expect(
                service.revokeSession('session-uuid', 'owner-uuid', 'admin'),
            ).rejects.toThrow(ForbiddenException);
        });

        it('should return false and not revoke if session is already revoked', async () => {
            const session = buildMockSession({ revokedAt: new Date() });
            sessionRepository.findById.mockResolvedValue(session);

            const result = await service.revokeSession('session-uuid');

            expect(result).toBe(false);
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
        });

        it('should revoke session and return true if active and matches', async () => {
            const session = buildMockSession({ userId: 'owner-uuid', realm: 'customer' });
            sessionRepository.findById.mockResolvedValue(session);
            sessionRepository.revoke.mockResolvedValue(true);

            const result = await service.revokeSession('session-uuid', 'owner-uuid', 'customer');

            expect(result).toBe(true);
            expect(sessionRepository.revoke).toHaveBeenCalledWith(
                'session-uuid',
                expect.any(Date),
                undefined,
            );
        });
    });

    describe('revokeCurrentSession', () => {
        it('should write Redis deny-list before revoking session, tokens, and recording logout event in transaction', async () => {
            const session = buildMockSession({
                id: 'session-uuid',
                userId: 'user-uuid',
                realm: 'customer',
                ipAddress: '198.51.100.10',
                userAgent: 'Stored UA',
            });
            sessionRepository.findById.mockResolvedValue(session);
            sessionRepository.revoke.mockResolvedValue(true);

            await service.revokeCurrentSession({
                sessionId: 'session-uuid',
                userId: 'user-uuid',
                realm: 'customer',
                accessTokenJti: 'access-jti',
                ipAddress: '203.0.113.1',
                userAgent: 'Request UA',
            });

            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('session-uuid', 930);
            expect(redisTokenService.revokeToken).toHaveBeenCalledWith('access-jti', 930);
            expect(transactionRepository.run).toHaveBeenCalledTimes(1);
            expect(sessionRepository.revoke).toHaveBeenCalledWith(
                'session-uuid',
                expect.any(Date),
                {},
            );
            expect(authTokenRepository.revokeTokensBySession).toHaveBeenCalledWith(
                'session-uuid',
                expect.any(Date),
                {},
            );
            expect(securityEventService.recordLogout).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    sessionId: 'session-uuid',
                    ipAddress: '203.0.113.1',
                    userAgent: 'Request UA',
                },
                {},
            );
        });

        it('should not start database transaction if Redis session revoke fails', async () => {
            const redisError = new Error('redis unavailable');
            redisTokenService.revokeSession.mockRejectedValue(redisError);

            await expect(
                service.revokeCurrentSession({
                    sessionId: 'session-uuid',
                    userId: 'user-uuid',
                    realm: 'customer',
                }),
            ).rejects.toBe(redisError);

            expect(transactionRepository.run).not.toHaveBeenCalled();
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
            expect(authTokenRepository.revokeTokensBySession).not.toHaveBeenCalled();
            expect(securityEventService.recordLogout).not.toHaveBeenCalled();
        });

        it('should throw ForbiddenException if session ownership does not match', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ userId: 'other-user', realm: 'customer' }),
            );

            await expect(
                service.revokeCurrentSession({
                    sessionId: 'session-uuid',
                    userId: 'user-uuid',
                    realm: 'customer',
                }),
            ).rejects.toThrow(ForbiddenException);

            expect(sessionRepository.revoke).not.toHaveBeenCalled();
            expect(authTokenRepository.revokeTokensBySession).not.toHaveBeenCalled();
            expect(securityEventService.recordLogout).not.toHaveBeenCalled();
        });
    });

    describe('revokeSpecificSession', () => {
        it('should write Redis deny-list before revoking session, tokens, and recording session_revoked event in transaction', async () => {
            const session = buildMockSession({
                id: 'session-uuid',
                userId: 'user-uuid',
                realm: 'customer',
            });
            sessionRepository.findById.mockResolvedValue(session);
            sessionRepository.revoke.mockResolvedValue(true);

            await service.revokeSpecificSession({
                sessionId: 'session-uuid',
                userId: 'user-uuid',
                realm: 'customer',
            });

            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('session-uuid', 930);
            expect(transactionRepository.run).toHaveBeenCalledTimes(1);
            expect(redisTokenService.revokeSession.mock.invocationCallOrder[0]).toBeLessThan(
                transactionRepository.run.mock.invocationCallOrder[0],
            );
            expect(sessionRepository.revoke).toHaveBeenCalledWith(
                'session-uuid',
                expect.any(Date),
                {},
            );
            expect(authTokenRepository.revokeTokensBySession).toHaveBeenCalledWith(
                'session-uuid',
                expect.any(Date),
                {},
            );
            expect(securityEventService.recordSessionRevoked).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    sessionId: 'session-uuid',
                    metadata: {
                        revocationReason: 'specific_session_revoke',
                    },
                },
                {},
            );
        });

        it('should not start database transaction if Redis session revoke fails', async () => {
            const redisError = new Error('redis unavailable');
            redisTokenService.revokeSession.mockRejectedValue(redisError);

            await expect(
                service.revokeSpecificSession({
                    sessionId: 'session-uuid',
                    userId: 'user-uuid',
                    realm: 'customer',
                }),
            ).rejects.toBe(redisError);

            expect(transactionRepository.run).not.toHaveBeenCalled();
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
            expect(authTokenRepository.revokeTokensBySession).not.toHaveBeenCalled();
            expect(securityEventService.recordSessionRevoked).not.toHaveBeenCalled();
        });

        it('should throw ForbiddenException if target session ownership does not match', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ userId: 'other-user', realm: 'customer' }),
            );

            await expect(
                service.revokeSpecificSession({
                    sessionId: 'session-uuid',
                    userId: 'user-uuid',
                    realm: 'customer',
                }),
            ).rejects.toThrow(ForbiddenException);

            expect(sessionRepository.revoke).not.toHaveBeenCalled();
            expect(authTokenRepository.revokeTokensBySession).not.toHaveBeenCalled();
            expect(securityEventService.recordSessionRevoked).not.toHaveBeenCalled();
        });
    });

    describe('revokeOtherSessions', () => {
        it('should revoke other sessions, tokens, and record an event', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedOtherByUserRealm.mockResolvedValue([
                buildMockSession({ id: 'other-session-1' }),
                buildMockSession({ id: 'other-session-2' }),
            ]);
            sessionRepository.revoke.mockResolvedValue(true);

            const result = await service.revokeOtherSessions({
                userId: 'user-uuid',
                realm: 'customer',
                currentSessionId: 'current-session-uuid',
            });

            expect(result).toBe(2);
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('other-session-1', 930);
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('other-session-2', 930);
            expect(sessionRepository.revoke).toHaveBeenCalledTimes(2);
            expect(authTokenRepository.revokeTokensBySession).toHaveBeenCalledTimes(2);
            expect(securityEventService.recordRevokeOtherSessions).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    sessionId: 'current-session-uuid',
                    metadata: {
                        revokedSessionCount: 2,
                    },
                },
                expect.any(Object),
            );
        });

        it('should enforce reauth on revokeOtherSessions when enabled', async () => {
            configService.get = jest.fn().mockImplementation((key: string) => {
                if (key === 'REAUTH_ENFORCEMENT_ENABLED') return true;
                return undefined;
            });

            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedOtherByUserRealm.mockResolvedValue([
                buildMockSession({ id: 'other-session-1' }),
            ]);

            reauthConfirmationService.consumeReauthConfirmation.mockResolvedValue(false);

            // Reauth fails
            await expect(
                service.revokeOtherSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                    reauthConfirmationToken: 'invalid-token',
                }),
            ).rejects.toThrow('Invalid or expired re-authentication confirmation token');

            // Token missing
            await expect(
                service.revokeOtherSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                }),
            ).rejects.toThrow('Re-authentication confirmation token is required');
        });

        it('should write Redis deny-list before opening DB transaction (Phase 25.2: fail-closed)', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedOtherByUserRealm.mockResolvedValue([
                buildMockSession({ id: 'other-session-1' }),
            ]);
            sessionRepository.revoke.mockResolvedValue(true);

            await service.revokeOtherSessions({
                userId: 'user-uuid',
                realm: 'customer',
                currentSessionId: 'current-session-uuid',
            });

            const redisCallOrder = redisTokenService.revokeSession.mock.invocationCallOrder[0];
            const transactionCallOrder = transactionRepository.run.mock.invocationCallOrder[0];
            expect(redisCallOrder).toBeLessThan(transactionCallOrder);
        });

        it('should not start DB transaction if Redis write fails (Phase 25.2: fail-closed)', async () => {
            const redisError = new Error('redis unavailable');
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedOtherByUserRealm.mockResolvedValue([
                buildMockSession({ id: 'other-session-1' }),
            ]);
            redisTokenService.revokeSession.mockRejectedValue(redisError);

            await expect(
                service.revokeOtherSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                }),
            ).rejects.toBe(redisError);

            expect(transactionRepository.run).not.toHaveBeenCalled();
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
            expect(authTokenRepository.revokeTokensBySession).not.toHaveBeenCalled();
            expect(securityEventService.recordRevokeOtherSessions).not.toHaveBeenCalled();
        });

        it('should not call Redis if current session not found (Phase 25.2: ownership before Redis)', async () => {
            sessionRepository.findById.mockResolvedValue(null);

            await expect(
                service.revokeOtherSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                }),
            ).rejects.toThrow(NotFoundException);

            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
            expect(transactionRepository.run).not.toHaveBeenCalled();
        });

        it('should include expired-but-unrevoked sessions in affected set (Phase 25.1)', async () => {
            const expiredSession = buildMockSession({
                id: 'expired-session-uuid',
                expiresAt: new Date(Date.now() - 1000), // expired 1 second ago
                revokedAt: null, // not revoked
            });
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedOtherByUserRealm.mockResolvedValue([expiredSession]);
            sessionRepository.revoke.mockResolvedValue(true);

            const result = await service.revokeOtherSessions({
                userId: 'user-uuid',
                realm: 'customer',
                currentSessionId: 'current-session-uuid',
            });

            expect(result).toBe(1);
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith(
                'expired-session-uuid',
                930,
            );
            expect(sessionRepository.revoke).toHaveBeenCalledWith(
                'expired-session-uuid',
                expect.any(Date),
                expect.any(Object),
            );
        });

        it('should not revoke current session (Phase 25.1: current not in affected set)', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            // findUnrevokedOtherByUserRealm correctly excludes current session
            sessionRepository.findUnrevokedOtherByUserRealm.mockResolvedValue([]);
            sessionRepository.revoke.mockResolvedValue(true);

            const result = await service.revokeOtherSessions({
                userId: 'user-uuid',
                realm: 'customer',
                currentSessionId: 'current-session-uuid',
            });

            expect(result).toBe(0);
            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
        });

        it('should reject revoke-other when current session ownership does not match', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid', userId: 'other-user' }),
            );

            await expect(
                service.revokeOtherSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                }),
            ).rejects.toThrow(ForbiddenException);

            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
        });
    });

    describe('revokeAllSessions', () => {
        it('should revoke all sessions, tokens, and record an event', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedByUserRealm.mockResolvedValue([
                buildMockSession({ id: 'current-session-uuid' }),
                buildMockSession({ id: 'other-session-1' }),
            ]);
            sessionRepository.revoke.mockResolvedValue(true);

            const result = await service.revokeAllSessions({
                userId: 'user-uuid',
                realm: 'customer',
                currentSessionId: 'current-session-uuid',
            });

            expect(result).toBe(2);
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith(
                'current-session-uuid',
                930,
            );
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('other-session-1', 930);
            expect(sessionRepository.revoke).toHaveBeenCalledTimes(2);
            expect(authTokenRepository.revokeTokensBySession).toHaveBeenCalledTimes(2);
            expect(securityEventService.recordRevokeAllSessions).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    sessionId: 'current-session-uuid',
                    metadata: {
                        revokedSessionCount: 2,
                        revocationReason: 'user_requested_all_sessions_revoke',
                    },
                },
                expect.any(Object),
            );
        });

        it('should write Redis deny-list before opening DB transaction (Phase 26.2: fail-closed)', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedByUserRealm.mockResolvedValue([
                buildMockSession({ id: 'current-session-uuid' }),
            ]);
            sessionRepository.revoke.mockResolvedValue(true);

            await service.revokeAllSessions({
                userId: 'user-uuid',
                realm: 'customer',
                currentSessionId: 'current-session-uuid',
            });

            const redisCallOrder = redisTokenService.revokeSession.mock.invocationCallOrder[0];
            const transactionCallOrder = transactionRepository.run.mock.invocationCallOrder[0];
            expect(redisCallOrder).toBeLessThan(transactionCallOrder);
        });

        it('should not start DB transaction if Redis write fails (Phase 26.2: fail-closed)', async () => {
            const redisError = new Error('redis unavailable');
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedByUserRealm.mockResolvedValue([
                buildMockSession({ id: 'current-session-uuid' }),
            ]);
            redisTokenService.revokeSession.mockRejectedValue(redisError);

            await expect(
                service.revokeAllSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                }),
            ).rejects.toBe(redisError);

            expect(transactionRepository.run).not.toHaveBeenCalled();
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
            expect(authTokenRepository.revokeTokensBySession).not.toHaveBeenCalled();
            expect(securityEventService.recordRevokeAllSessions).not.toHaveBeenCalled();
        });

        it('should not call Redis if current session not found (Phase 26.2: ownership before Redis)', async () => {
            sessionRepository.findById.mockResolvedValue(null);

            await expect(
                service.revokeAllSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                }),
            ).rejects.toThrow(NotFoundException);

            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
            expect(transactionRepository.run).not.toHaveBeenCalled();
        });

        it('should reject when current session ownership does not match', async () => {
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid', userId: 'other-user' }),
            );

            await expect(
                service.revokeAllSessions({
                    userId: 'user-uuid',
                    realm: 'customer',
                    currentSessionId: 'current-session-uuid',
                }),
            ).rejects.toThrow(ForbiddenException);

            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
            expect(sessionRepository.revoke).not.toHaveBeenCalled();
        });

        it('should include expired-but-unrevoked sessions in affected set', async () => {
            const expiredSession = buildMockSession({
                id: 'expired-session-uuid',
                expiresAt: new Date(Date.now() - 1000),
                revokedAt: null,
            });
            sessionRepository.findById.mockResolvedValue(
                buildMockSession({ id: 'current-session-uuid' }),
            );
            sessionRepository.findUnrevokedByUserRealm.mockResolvedValue([expiredSession]);
            sessionRepository.revoke.mockResolvedValue(true);

            const result = await service.revokeAllSessions({
                userId: 'user-uuid',
                realm: 'customer',
                currentSessionId: 'current-session-uuid',
            });

            expect(result).toBe(1);
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith(
                'expired-session-uuid',
                930,
            );
        });
    });

    describe('revokeUserSessions', () => {
        it('should revoke all active user sessions and record password reset events by realm', async () => {
            sessionRepository.findActiveByUser.mockResolvedValue([
                buildMockSession({ id: 'customer-session', realm: 'customer' }),
                buildMockSession({ id: 'admin-session', realm: 'admin' }),
            ]);
            sessionRepository.revoke.mockResolvedValue(true);

            const result = await service.revokeUserSessions({
                userId: 'user-uuid',
                eventType: SecurityEventType.PASSWORD_RESET_COMPLETED,
            });

            expect(result).toBe(2);
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('customer-session', 930);
            expect(redisTokenService.revokeSession).toHaveBeenCalledWith('admin-session', 930);
            expect(authTokenRepository.revokeTokensBySession).toHaveBeenCalledTimes(2);
            expect(securityEventService.recordPasswordResetCompleted).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    metadata: {
                        revokedSessionCount: 1,
                    },
                },
                expect.any(Object),
            );
            expect(securityEventService.recordPasswordResetCompleted).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'admin',
                    metadata: {
                        revokedSessionCount: 1,
                    },
                },
                expect.any(Object),
            );
        });

        it('should record password reset event when there are no active sessions', async () => {
            sessionRepository.findActiveByUser.mockResolvedValue([]);

            const result = await service.revokeUserSessions({
                userId: 'user-uuid',
                eventType: SecurityEventType.PASSWORD_RESET_COMPLETED,
                fallbackRealm: 'customer',
            });

            expect(result).toBe(0);
            expect(redisTokenService.revokeSession).not.toHaveBeenCalled();
            expect(authTokenRepository.revokeTokensBySession).not.toHaveBeenCalled();
            expect(securityEventService.recordPasswordResetCompleted).toHaveBeenCalledWith(
                {
                    userId: 'user-uuid',
                    realm: 'customer',
                    metadata: {
                        revokedSessionCount: 0,
                    },
                },
                expect.any(Object),
            );
        });
    });

    describe('listActiveSessions', () => {
        it('should call findActiveByUserRealm and return sessions list', async () => {
            const list = [buildMockSession()];
            sessionRepository.findActiveByUserRealm.mockResolvedValue(list);

            const result = await service.listActiveSessions('user-uuid', 'customer');

            expect(result).toEqual(list);
            expect(sessionRepository.findActiveByUserRealm).toHaveBeenCalledWith(
                'user-uuid',
                'customer',
                undefined,
            );
        });
    });
});
