import { BadRequestException } from '@nestjs/common';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { RefreshTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { RefreshTokenVerificationService } from '@/module-auth-token/services/refresh-token-verification.service';
import { TokenService } from '@/module-auth-token/services/token.service';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';

describe('RefreshTokenVerificationService', () => {
    let service: RefreshTokenVerificationService;
    let tokenService: {
        verifyToken: jest.MockedFunction<TokenService['verifyToken']>;
    };
    let repository: {
        findRefreshTokenById: jest.MockedFunction<AuthTokenRepository['findRefreshTokenById']>;
    };
    let sessionRepository: {
        findById: jest.MockedFunction<SessionRepository['findById']>;
    };

    const refreshTokenPayload: RefreshTokenPayload = {
        sub: 'user-id',
        userId: 'user-id',
        type: TokenType.REFRESH,
        sessionId: 'session-id',
        jti: 'refresh-jti',
        tokenId: 'refresh-token-id',
        ipAddress: '203.0.113.1',
        userAgent: 'mac chrome',
    };

    beforeEach(() => {
        tokenService = {
            verifyToken: jest.fn(),
        };
        repository = {
            findRefreshTokenById: jest.fn(),
        };
        sessionRepository = {
            findById: jest.fn().mockResolvedValue({
                id: 'session-id',
                userId: 'user-id',
                realm: 'customer',
                knownDeviceId: 'known-device-id',
                deviceId: 'device-id',
                ipAddress: '127.0.0.1',
                userAgent: 'test-agent',
                createdAt: new Date(),
                updatedAt: new Date(),
                lastSeenAt: new Date(),
                expiresAt: new Date(Date.now() + 600_000),
                revokedAt: null,
                lastCountry: null,
                lastRegion: null,
                lastCity: null,
                riskScore: 0,
                riskReason: null,
            }),
        };

        service = new RefreshTokenVerificationService(
            tokenService as unknown as TokenService,
            repository as unknown as AuthTokenRepository,
            sessionRepository as unknown as SessionRepository,
        );
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    it('loads token metadata with user data and returns verified user on hash match', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue({
            tokenId: refreshTokenPayload.tokenId,
            sessionId: refreshTokenPayload.sessionId,
            jti: refreshTokenPayload.jti,
            refreshToken: 'stored-hash',
            replacedByTokenId: null,
            replacedAt: null,
            graceExpiresAt: null,
            replacementSessionId: null,
            replacementRevokedAt: null,
            ipAddress: refreshTokenPayload.ipAddress,
            userAgent: 'mac chrome',
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: null,
        });
        jest.spyOn(Argon2HashUtil, 'compare').mockResolvedValue(true);

        const result = await service.verify('raw-token');

        expect(repository.findRefreshTokenById).toHaveBeenCalledWith(refreshTokenPayload.tokenId);
        expect(result).toEqual({
            verified: true,
            tokenId: refreshTokenPayload.tokenId,
            sessionId: refreshTokenPayload.sessionId,
            isGrace: false,
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
        });
    });

    it('returns false when stored token metadata is missing', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue(null);

        await expect(service.verify('raw-token')).resolves.toEqual({ verified: false });
    });

    it('returns false when session identity does not match', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue({
            tokenId: refreshTokenPayload.tokenId,
            sessionId: 'different-session-id',
            jti: refreshTokenPayload.jti,
            refreshToken: 'stored-hash',
            replacedByTokenId: null,
            replacedAt: null,
            graceExpiresAt: null,
            replacementSessionId: null,
            replacementRevokedAt: null,
            ipAddress: refreshTokenPayload.ipAddress,
            userAgent: 'mac chrome',
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: null,
        });

        jest.spyOn(Argon2HashUtil, 'compare').mockResolvedValue(true);

        await expect(service.verify('raw-token')).resolves.toEqual({ verified: false });
    });

    it('returns false when hash comparison fails', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue({
            tokenId: refreshTokenPayload.tokenId,
            sessionId: refreshTokenPayload.sessionId,
            jti: refreshTokenPayload.jti,
            refreshToken: 'stored-hash',
            replacedByTokenId: null,
            replacedAt: null,
            graceExpiresAt: null,
            replacementSessionId: null,
            replacementRevokedAt: null,
            ipAddress: refreshTokenPayload.ipAddress,
            userAgent: 'mac chrome',
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: null,
        });
        jest.spyOn(Argon2HashUtil, 'compare').mockResolvedValue(false);

        await expect(service.verify('raw-token')).resolves.toEqual({ verified: false });
    });

    it('throws when jwt payload is missing', async () => {
        tokenService.verifyToken.mockReturnValue(null as never);

        await expect(service.verify('raw-token')).rejects.toBeInstanceOf(BadRequestException);
    });

    it('accepts a replaced token only inside grace and for the same session', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue({
            tokenId: refreshTokenPayload.tokenId,
            sessionId: refreshTokenPayload.sessionId,
            jti: refreshTokenPayload.jti,
            refreshToken: 'stored-hash',
            replacedByTokenId: 'replacement-token-id',
            replacedAt: new Date(),
            graceExpiresAt: new Date(Date.now() + 10_000),
            replacementSessionId: refreshTokenPayload.sessionId,
            replacementRevokedAt: null,
            ipAddress: refreshTokenPayload.ipAddress,
            userAgent: refreshTokenPayload.userAgent,
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: null,
        });
        jest.spyOn(Argon2HashUtil, 'compare').mockResolvedValue(true);

        await expect(service.verify('raw-token')).resolves.toEqual(
            expect.objectContaining({
                verified: true,
                tokenId: refreshTokenPayload.tokenId,
            }),
        );
    });

    it('rejects an expired or cross-session grace token', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue({
            tokenId: refreshTokenPayload.tokenId,
            sessionId: refreshTokenPayload.sessionId,
            jti: refreshTokenPayload.jti,
            refreshToken: 'stored-hash',
            replacedByTokenId: 'replacement-token-id',
            replacedAt: new Date(),
            graceExpiresAt: new Date(Date.now() - 1),
            replacementSessionId: 'different-session-id',
            replacementRevokedAt: null,
            ipAddress: refreshTokenPayload.ipAddress,
            userAgent: refreshTokenPayload.userAgent,
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: null,
        });

        await expect(service.verify('raw-token')).resolves.toEqual({ verified: false });
    });

    it('rejects a revoked token row', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue({
            tokenId: refreshTokenPayload.tokenId,
            sessionId: refreshTokenPayload.sessionId,
            jti: refreshTokenPayload.jti,
            refreshToken: 'stored-hash',
            replacedByTokenId: null,
            replacedAt: null,
            graceExpiresAt: null,
            replacementSessionId: null,
            replacementRevokedAt: null,
            ipAddress: refreshTokenPayload.ipAddress,
            userAgent: 'mac chrome',
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: new Date(),
        });

        await expect(service.verify('raw-token')).resolves.toEqual({ verified: false });
    });

    it('rejects an expired token row', async () => {
        tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
        repository.findRefreshTokenById.mockResolvedValue({
            tokenId: refreshTokenPayload.tokenId,
            sessionId: refreshTokenPayload.sessionId,
            jti: refreshTokenPayload.jti,
            refreshToken: 'stored-hash',
            replacedByTokenId: null,
            replacedAt: null,
            graceExpiresAt: null,
            replacementSessionId: null,
            replacementRevokedAt: null,
            ipAddress: refreshTokenPayload.ipAddress,
            userAgent: 'mac chrome',
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() - 1000),
            revokedAt: null,
        });

        await expect(service.verify('raw-token')).resolves.toEqual({ verified: false });
    });

    describe('Individual Predicates', () => {
        const tokenBase = {
            tokenId: 'token-id',
            sessionId: 'session-id',
            jti: 'jti',
            refreshToken: 'hash',
            ipAddress: '127.0.0.1',
            userAgent: 'test',
            user: {
                id: 'user-id',
                name: 'Test User',
                email: 'test@example.com',
                role: UserRoles.USER,
            },
        };

        describe('isCurrentToken', () => {
            it('returns true for active, non-replaced, non-revoked token', () => {
                const token = {
                    ...tokenBase,
                    replacedByTokenId: null,
                    replacedAt: null,
                    graceExpiresAt: null,
                    replacementSessionId: null,
                    replacementRevokedAt: null,
                    expiresAt: new Date(Date.now() + 10_000),
                    revokedAt: null,
                };
                expect(service.isCurrentToken(token)).toBe(true);
            });

            it('returns false for replaced token', () => {
                const token = {
                    ...tokenBase,
                    replacedByTokenId: 'new-id',
                    replacedAt: new Date(),
                    graceExpiresAt: new Date(Date.now() + 10_000),
                    replacementSessionId: 'session-id',
                    replacementRevokedAt: null,
                    expiresAt: new Date(Date.now() + 10_000),
                    revokedAt: null,
                };
                expect(service.isCurrentToken(token)).toBe(false);
            });
        });

        describe('isValidGraceToken', () => {
            it('returns true for replaced token within grace period', () => {
                const token = {
                    ...tokenBase,
                    replacedByTokenId: 'new-id',
                    replacedAt: new Date(),
                    graceExpiresAt: new Date(Date.now() + 10_000),
                    replacementSessionId: 'session-id',
                    replacementRevokedAt: null,
                    expiresAt: new Date(Date.now() + 10_000),
                    revokedAt: null,
                };
                expect(service.isValidGraceToken(token)).toBe(true);
            });

            it('returns false if grace has expired', () => {
                const token = {
                    ...tokenBase,
                    replacedByTokenId: 'new-id',
                    replacedAt: new Date(),
                    graceExpiresAt: new Date(Date.now() - 1000),
                    replacementSessionId: 'session-id',
                    replacementRevokedAt: null,
                    expiresAt: new Date(Date.now() + 10_000),
                    revokedAt: null,
                };
                expect(service.isValidGraceToken(token)).toBe(false);
            });
        });

        describe('isRevokedToken', () => {
            it('returns true if revokedAt is set', () => {
                const token = {
                    ...tokenBase,
                    replacedByTokenId: null,
                    replacedAt: null,
                    graceExpiresAt: null,
                    replacementSessionId: null,
                    replacementRevokedAt: null,
                    expiresAt: new Date(Date.now() + 10_000),
                    revokedAt: new Date(),
                };
                expect(service.isRevokedToken(token)).toBe(true);
            });
        });

        describe('isExpiredToken', () => {
            it('returns true if expiresAt is in the past', () => {
                const token = {
                    ...tokenBase,
                    replacedByTokenId: null,
                    replacedAt: null,
                    graceExpiresAt: null,
                    replacementSessionId: null,
                    replacementRevokedAt: null,
                    expiresAt: new Date(Date.now() - 1000),
                    revokedAt: null,
                };
                expect(service.isExpiredToken(token)).toBe(true);
            });
        });
    });

    describe('Session and Realm Validation', () => {
        beforeEach(() => {
            tokenService.verifyToken.mockReturnValue(refreshTokenPayload);
            repository.findRefreshTokenById.mockResolvedValue({
                tokenId: refreshTokenPayload.tokenId,
                sessionId: refreshTokenPayload.sessionId,
                jti: refreshTokenPayload.jti,
                refreshToken: 'stored-hash',
                replacedByTokenId: null,
                replacedAt: null,
                graceExpiresAt: null,
                replacementSessionId: null,
                replacementRevokedAt: null,
                ipAddress: refreshTokenPayload.ipAddress,
                userAgent: 'mac chrome',
                user: {
                    id: 'user-id',
                    name: 'John Doe',
                    email: 'john@example.com',
                    role: UserRoles.USER,
                },
                expiresAt: new Date(Date.now() + 600_000),
                revokedAt: null,
            });
            jest.spyOn(Argon2HashUtil, 'compare').mockResolvedValue(true);
        });

        it('returns false when session is missing', async () => {
            sessionRepository.findById.mockResolvedValue(null);
            const result = await service.verify('raw-token');
            expect(result).toEqual({ verified: false });
        });

        it('returns false when session is revoked', async () => {
            sessionRepository.findById.mockResolvedValue({
                id: 'session-id',
                userId: 'user-id',
                realm: 'customer',
                knownDeviceId: 'known-device-id',
                deviceId: 'device-id',
                ipAddress: '127.0.0.1',
                userAgent: 'test-agent',
                createdAt: new Date(),
                updatedAt: new Date(),
                lastSeenAt: new Date(),
                expiresAt: new Date(Date.now() + 600_000),
                revokedAt: new Date(),
                lastCountry: null,
                lastRegion: null,
                lastCity: null,
                riskScore: 0,
                riskReason: null,
            });
            const result = await service.verify('raw-token');
            expect(result).toEqual({ verified: false });
        });

        it('returns false when session is expired', async () => {
            sessionRepository.findById.mockResolvedValue({
                id: 'session-id',
                userId: 'user-id',
                realm: 'customer',
                knownDeviceId: 'known-device-id',
                deviceId: 'device-id',
                ipAddress: '127.0.0.1',
                userAgent: 'test-agent',
                createdAt: new Date(),
                updatedAt: new Date(),
                lastSeenAt: new Date(),
                expiresAt: new Date(Date.now() - 1000),
                revokedAt: null,
                lastCountry: null,
                lastRegion: null,
                lastCity: null,
                riskScore: 0,
                riskReason: null,
            });
            const result = await service.verify('raw-token');
            expect(result).toEqual({ verified: false });
        });

        it('returns false when expectedRealm mismatches', async () => {
            sessionRepository.findById.mockResolvedValue({
                id: 'session-id',
                userId: 'user-id',
                realm: 'customer',
                knownDeviceId: 'known-device-id',
                deviceId: 'device-id',
                ipAddress: '127.0.0.1',
                userAgent: 'test-agent',
                createdAt: new Date(),
                updatedAt: new Date(),
                lastSeenAt: new Date(),
                expiresAt: new Date(Date.now() + 600_000),
                revokedAt: null,
                lastCountry: null,
                lastRegion: null,
                lastCity: null,
                riskScore: 0,
                riskReason: null,
            });
            const result = await service.verify('raw-token', 'admin');
            expect(result).toEqual({ verified: false });
        });

        it('returns true when expectedRealm matches', async () => {
            sessionRepository.findById.mockResolvedValue({
                id: 'session-id',
                userId: 'user-id',
                realm: 'customer',
                knownDeviceId: 'known-device-id',
                deviceId: 'device-id',
                ipAddress: '127.0.0.1',
                userAgent: 'test-agent',
                createdAt: new Date(),
                updatedAt: new Date(),
                lastSeenAt: new Date(),
                expiresAt: new Date(Date.now() + 600_000),
                revokedAt: null,
                lastCountry: null,
                lastRegion: null,
                lastCity: null,
                riskScore: 0,
                riskReason: null,
            });
            const result = await service.verify('raw-token', 'customer');
            expect(result.verified).toBe(true);
        });
    });
});
