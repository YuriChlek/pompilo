import { BadRequestException, InternalServerErrorException } from '@nestjs/common';
import { AuthTokenService } from '@/module-auth-token/services/auth-token.service';
import { TokenService } from '@/module-auth-token/services/token.service';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { UserJwtPayload } from '@/module-user/interfaces/user.interfaces';
import { UserRoles } from '@/module-auth/enums/auth.enums';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { AuthTokenPayloadService } from '@/module-auth-token/services/auth-token-payload.service';
import { RefreshTokenVerificationService } from '@/module-auth-token/services/refresh-token-verification.service';
import { RefreshTokenStorageService } from '@/module-auth-token/services/refresh-token-storage.service';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';
import { ConfigService } from '@nestjs/config';
import { TokenFromDbPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { SessionSelect } from '@/module-auth-token/schemas/sessions.schema';
import { EncryptService } from '@/module-encrypt/services/encrypt.service';

describe('AuthTokenService', () => {
    let service: AuthTokenService;
    let tokenService: {
        createAccessToken: jest.MockedFunction<TokenService['createAccessToken']>;
        createRefreshToken: jest.MockedFunction<TokenService['createRefreshToken']>;
        verifyToken: jest.MockedFunction<TokenService['verifyToken']>;
    };
    let repository: {
        removeRefreshToken: jest.MockedFunction<AuthTokenRepository['removeRefreshToken']>;
        findRefreshTokenById: jest.MockedFunction<AuthTokenRepository['findRefreshTokenById']>;
        rotateToken: jest.MockedFunction<AuthTokenRepository['rotateToken']>;
        revokeTokensBySession: jest.MockedFunction<AuthTokenRepository['revokeTokensBySession']>;
    };
    let redisTokenService: {
        revokeToken: jest.MockedFunction<RedisTokenService['revokeToken']>;
        revokeSession: jest.MockedFunction<RedisTokenService['revokeSession']>;
    };
    let configService: {
        getOrThrow: jest.MockedFunction<ConfigService['getOrThrow']>;
    };
    let payloadService: {
        createAccessTokenPayload: jest.MockedFunction<
            AuthTokenPayloadService['createAccessTokenPayload']
        >;
        createRefreshTokenPayload: jest.MockedFunction<
            AuthTokenPayloadService['createRefreshTokenPayload']
        >;
    };
    let refreshTokenVerificationService: {
        verify: jest.MockedFunction<RefreshTokenVerificationService['verify']>;
        isValidGraceToken: jest.MockedFunction<
            RefreshTokenVerificationService['isValidGraceToken']
        >;
    };
    let refreshTokenStorageService: {
        saveRefreshToken: jest.MockedFunction<RefreshTokenStorageService['saveRefreshToken']>;
    };
    let encryptService: {
        encrypt: jest.MockedFunction<EncryptService['encrypt']>;
        decrypt: jest.MockedFunction<EncryptService['decrypt']>;
    };

    const userPayload: UserJwtPayload = {
        id: 'user-id',
        name: 'John Doe',
        email: 'john@example.com',
        role: UserRoles.USER,
        ipAddress: '203.0.113.1',
        userAgent: 'Mac Chrome',
        sessionId: 'session-id',
    };

    beforeEach(() => {
        tokenService = {
            createAccessToken: jest.fn().mockReturnValue('signed-access'),
            createRefreshToken: jest.fn().mockReturnValue('signed-refresh'),
            verifyToken: jest.fn(),
        };

        repository = {
            removeRefreshToken: jest.fn(),
            findRefreshTokenById: jest.fn(),
            rotateToken: jest.fn(),
            revokeTokensBySession: jest.fn(),
        };

        payloadService = {
            createAccessTokenPayload: jest.fn().mockReturnValue({
                sub: userPayload.id,
                userId: userPayload.id,
                sessionId: userPayload.sessionId,
                email: userPayload.email,
                username: userPayload.name,
                role: userPayload.role,
                realm: 'customer',
                ipAddress: userPayload.ipAddress,
                userAgent: 'mac chrome',
                type: TokenType.ACCESS,
                jti: 'mock-access-jti',
            }),
            createRefreshTokenPayload: jest.fn().mockImplementation((user: UserJwtPayload) => ({
                sub: user.id,
                userId: user.id,
                sessionId: user.sessionId,
                userAgent: 'mac chrome',
                ipAddress: user.ipAddress,
                type: TokenType.REFRESH,
                jti: 'refresh-jti',
                tokenId: 'refresh-token-id',
            })),
        };

        refreshTokenVerificationService = {
            verify: jest.fn(),
            isValidGraceToken: jest.fn(),
        };
        refreshTokenStorageService = {
            saveRefreshToken: jest.fn(),
        };
        redisTokenService = {
            revokeToken: jest.fn(),
            revokeSession: jest.fn(),
        };
        configService = {
            getOrThrow: jest.fn().mockReturnValue('30m'),
        };
        encryptService = {
            encrypt: jest.fn().mockReturnValue('encrypted-replacement'),
            decrypt: jest.fn().mockReturnValue('decrypted-replacement'),
        };

        service = new AuthTokenService(
            tokenService as unknown as TokenService,
            repository as unknown as AuthTokenRepository,
            payloadService as unknown as AuthTokenPayloadService,
            refreshTokenVerificationService as unknown as RefreshTokenVerificationService,
            refreshTokenStorageService as unknown as RefreshTokenStorageService,
            redisTokenService as unknown as RedisTokenService,
            configService as unknown as ConfigService,
            encryptService as unknown as EncryptService,
        );
        jest.clearAllMocks();
    });

    describe('createAccessToken', () => {
        it('delegates to tokenService with mapped payload', () => {
            const token = service.createAccessToken(userPayload);

            expect(payloadService.createAccessTokenPayload).toHaveBeenCalledWith(userPayload);
            expect(tokenService.createAccessToken).toHaveBeenCalledWith(
                expect.objectContaining({
                    sub: userPayload.id,
                    type: TokenType.ACCESS,
                }),
            );
            expect(token).toBe('signed-access');
        });

        it('wraps unexpected failures into InternalServerErrorException', () => {
            tokenService.createAccessToken.mockImplementation(() => {
                throw new Error('failed');
            });

            expect(() => service.createAccessToken(userPayload)).toThrow(
                InternalServerErrorException,
            );
        });
    });

    describe('createRefreshToken', () => {
        it('persists refresh token metadata', async () => {
            tokenService.createRefreshToken.mockReturnValue('refresh-token');

            const token = await service.createRefreshToken(userPayload);

            expect(token).toBe('refresh-token');
            expect(payloadService.createRefreshTokenPayload).toHaveBeenCalledWith(userPayload);
            expect(refreshTokenStorageService.saveRefreshToken).toHaveBeenCalledWith(
                expect.objectContaining({
                    userId: userPayload.id,
                    sessionId: userPayload.sessionId,
                    ipAddress: userPayload.ipAddress,
                    userAgent: 'mac chrome',
                }),
                'refresh-token',
            );
        });

        it('throws InternalServerErrorException on persistence errors', async () => {
            refreshTokenStorageService.saveRefreshToken.mockRejectedValue(new Error('db error'));

            await expect(service.createRefreshToken(userPayload)).rejects.toBeInstanceOf(
                InternalServerErrorException,
            );
        });
    });

    describe('verifyRefreshToken', () => {
        it('returns user info when stored token matches', async () => {
            refreshTokenVerificationService.verify.mockResolvedValue({
                verified: true,
                tokenId: 'refresh-token-id',
                sessionId: 'session-id',
                user: {
                    id: 'user-id',
                    email: 'john@example.com',
                    name: 'John Doe',
                    role: UserRoles.USER,
                },
                isGrace: false,
            });

            const result = await service.verifyRefreshToken('token', 'customer');

            expect(result).toEqual({
                verified: true,
                tokenId: 'refresh-token-id',
                sessionId: 'session-id',
                isGrace: false,
                user: {
                    id: 'user-id',
                    email: 'john@example.com',
                    name: 'John Doe',
                    role: 'user',
                },
            });
            expect(refreshTokenVerificationService.verify).toHaveBeenCalledWith(
                'token',
                'customer',
            );
        });

        it('rethrows verification errors', async () => {
            refreshTokenVerificationService.verify.mockRejectedValue(
                new BadRequestException('Refresh token not found.'),
            );

            await expect(service.verifyRefreshToken('token')).rejects.toBeInstanceOf(
                BadRequestException,
            );
        });

        it('returns false when token metadata not found', async () => {
            refreshTokenVerificationService.verify.mockResolvedValue({ verified: false });

            const result = await service.verifyRefreshToken('token');

            expect(result).toEqual({ verified: false });
        });
    });

    it('removes refresh tokens by session id', async () => {
        await service.removeRefreshToken('session-id');

        expect(repository.removeRefreshToken).toHaveBeenCalledWith('session-id');
    });

    it('revokes all refresh tokens in a session', async () => {
        await service.revokeTokensBySession('session-id');

        expect(repository.revokeTokensBySession).toHaveBeenCalledWith(
            'session-id',
            expect.any(Date),
            undefined,
        );
    });

    it('revokes access token and session ids in Redis', async () => {
        await service.revokeAccessSession({
            sub: 'user-id',
            userId: 'user-id',
            sessionId: 'session-id',
            email: 'john@example.com',
            username: 'John Doe',
            role: UserRoles.USER,
            realm: 'customer',
            ipAddress: '203.0.113.1',
            userAgent: 'mac chrome',
            type: TokenType.ACCESS,
            jti: 'access-jti',
        });

        expect(redisTokenService.revokeToken).toHaveBeenCalledWith('access-jti', 1800);
        expect(redisTokenService.revokeSession).toHaveBeenCalledWith('session-id', 1800);
    });

    describe('createInitialRefreshToken', () => {
        it('should map the custom sessionId and persist refresh token metadata', async () => {
            tokenService.createRefreshToken.mockReturnValue('initial-refresh-token');

            const token = await service.createInitialRefreshToken('custom-session-id', userPayload);

            expect(token).toBe('initial-refresh-token');
            expect(payloadService.createRefreshTokenPayload).toHaveBeenCalledWith(
                expect.objectContaining({
                    id: userPayload.id,
                    sessionId: 'custom-session-id',
                }),
            );
            expect(refreshTokenStorageService.saveRefreshToken).toHaveBeenCalledWith(
                expect.objectContaining({
                    userId: userPayload.id,
                    sessionId: 'custom-session-id',
                }),
                'initial-refresh-token',
                undefined,
            );
        });

        it('throws InternalServerErrorException on persistence errors', async () => {
            refreshTokenStorageService.saveRefreshToken.mockRejectedValue(new Error('db error'));

            await expect(
                service.createInitialRefreshToken('custom-session-id', userPayload),
            ).rejects.toBeInstanceOf(InternalServerErrorException);
        });
    });

    describe('rotateRefreshToken', () => {
        const tokenRow: TokenFromDbPayload = {
            tokenId: 'old-token-id',
            sessionId: 'session-id',
            jti: 'old-jti',
            refreshToken: 'old-hash',
            replacedByTokenId: null,
            replacedAt: null,
            graceExpiresAt: null,
            replacementSessionId: null,
            replacementRevokedAt: null,
            ipAddress: '203.0.113.1',
            userAgent: 'Mac Chrome',
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: null,
        };

        const session: SessionSelect = {
            id: 'session-id',
            userId: 'user-id',
            realm: 'customer',
            knownDeviceId: 'device-id',
            deviceId: 'device-uuid',
            ipAddress: '203.0.113.1',
            userAgent: 'Mac Chrome',
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
        };

        it('should rotate token successfully and return raw new refresh token', async () => {
            const mockNewTokenModel = {
                id: 'new-token-id',
                sessionId: 'session-id',
                jti: 'new-jti',
                createdAt: new Date(),
                updatedAt: new Date(),
                refreshTokenHash: 'new-hash',
                encryptedReplacementToken: 'encrypted-replacement',
                expiresAt: new Date(),
                revokedAt: null,
                replacedByTokenId: null,
                replacedAt: null,
                graceExpiresAt: null,
            };

            repository.rotateToken.mockResolvedValue(mockNewTokenModel);
            tokenService.createRefreshToken.mockReturnValue('new-raw-refresh-token');
            encryptService.encrypt.mockReturnValue('encrypted-replacement');

            const result = await service.rotateRefreshToken(tokenRow, session);

            expect(result).toBe('new-raw-refresh-token');
            expect(encryptService.encrypt).toHaveBeenCalledWith('new-raw-refresh-token');
            expect(repository.rotateToken).toHaveBeenCalledWith(
                'old-token-id',
                expect.objectContaining({
                    sessionId: 'session-id',
                }),
                expect.any(Date),
                'encrypted-replacement',
                undefined,
            );
        });

        it('should return null if repository rotation returns null (concurrency conflict)', async () => {
            repository.rotateToken.mockResolvedValue(null);

            const result = await service.rotateRefreshToken(tokenRow, session);

            expect(result).toBeNull();
        });

        it('throws InternalServerErrorException on database error', async () => {
            repository.rotateToken.mockRejectedValue(new Error('db error'));

            await expect(service.rotateRefreshToken(tokenRow, session)).rejects.toBeInstanceOf(
                InternalServerErrorException,
            );
        });
    });

    describe('acceptReplacedTokenInGrace', () => {
        const tokenRow: TokenFromDbPayload = {
            tokenId: 'old-token-id',
            sessionId: 'session-id',
            jti: 'old-jti',
            refreshToken: 'old-hash',
            replacedByTokenId: 'new-token-id',
            replacedAt: new Date(),
            graceExpiresAt: new Date(Date.now() + 10_000),
            replacementSessionId: 'session-id',
            replacementRevokedAt: null,
            ipAddress: '203.0.113.1',
            userAgent: 'Mac Chrome',
            user: {
                id: 'user-id',
                name: 'John Doe',
                email: 'john@example.com',
                role: UserRoles.USER,
            },
            expiresAt: new Date(Date.now() + 600_000),
            revokedAt: null,
            encryptedReplacementToken: 'encrypted-replacement-token',
        };

        it('returns decrypted replacement token when inside grace period', async () => {
            refreshTokenVerificationService.isValidGraceToken.mockReturnValue(true);
            encryptService.decrypt.mockReturnValue('raw-replacement-token');

            const result = await service.acceptReplacedTokenInGrace(tokenRow);

            expect(result).toBe('raw-replacement-token');
            expect(refreshTokenVerificationService.isValidGraceToken).toHaveBeenCalledWith(
                tokenRow,
                expect.any(Date),
            );
            expect(encryptService.decrypt).toHaveBeenCalledWith('encrypted-replacement-token');
        });

        it('returns null when not inside grace period', async () => {
            refreshTokenVerificationService.isValidGraceToken.mockReturnValue(false);

            const result = await service.acceptReplacedTokenInGrace(tokenRow);

            expect(result).toBeNull();
        });

        it('returns null if encryptedReplacementToken is missing', async () => {
            refreshTokenVerificationService.isValidGraceToken.mockReturnValue(true);
            const tokenRowWithoutEncryption = { ...tokenRow, encryptedReplacementToken: null };

            const result = await service.acceptReplacedTokenInGrace(tokenRowWithoutEncryption);

            expect(result).toBeNull();
        });

        it('throws InternalServerErrorException on decryption error', async () => {
            refreshTokenVerificationService.isValidGraceToken.mockReturnValue(true);
            encryptService.decrypt.mockImplementation(() => {
                throw new Error('decryption failed');
            });

            await expect(service.acceptReplacedTokenInGrace(tokenRow)).rejects.toBeInstanceOf(
                InternalServerErrorException,
            );
        });

        it('loads token row by id before accepting grace replacement', async () => {
            repository.findRefreshTokenById.mockResolvedValue(tokenRow);
            refreshTokenVerificationService.isValidGraceToken.mockReturnValue(true);
            encryptService.decrypt.mockReturnValue('raw-replacement-token');

            const result = await service.acceptReplacedTokenInGraceById('old-token-id');

            expect(result).toBe('raw-replacement-token');
            expect(repository.findRefreshTokenById).toHaveBeenCalledWith('old-token-id');
            expect(encryptService.decrypt).toHaveBeenCalledWith('encrypted-replacement-token');
        });
    });
});
