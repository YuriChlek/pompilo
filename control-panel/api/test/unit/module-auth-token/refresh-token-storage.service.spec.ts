import { ConfigService } from '@nestjs/config';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';
import { RefreshTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { AuthTokenRepository } from '@/module-auth-token/repository/auth-token.repository';
import { SessionRepository } from '@/module-auth-token/repository/session.repository';
import { RefreshTokenStorageService } from '@/module-auth-token/services/refresh-token-storage.service';
import { Argon2HashUtil } from '@/common/utils/hash.util';
import { SessionSelect } from '@/module-auth-token/schemas/sessions.schema';

describe('RefreshTokenStorageService', () => {
    let service: RefreshTokenStorageService;
    let repository: {
        createRefreshToken: jest.MockedFunction<AuthTokenRepository['createRefreshToken']>;
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
        ipAddress: '203.0.113.1',
        userAgent: 'Mac Chrome',
        tokenId: 'refresh-token-id',
    };

    beforeEach(() => {
        repository = {
            createRefreshToken: jest.fn(),
        };

        sessionRepository = {
            findById: jest.fn().mockResolvedValue(null),
        };

        const configService = {
            getOrThrow: jest.fn().mockReturnValue('7d'),
        } as unknown as ConfigService;

        service = new RefreshTokenStorageService(
            repository as unknown as AuthTokenRepository,
            sessionRepository as unknown as SessionRepository,
            configService,
        );
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    it('creates refresh token metadata keyed by tokenId', async () => {
        jest.spyOn(Argon2HashUtil, 'hash').mockResolvedValue('hashed-token');

        await service.saveRefreshToken(refreshTokenPayload, 'raw-token');

        expect(repository.createRefreshToken).toHaveBeenCalledWith(
            expect.objectContaining({
                tokenId: refreshTokenPayload.tokenId,
                jti: refreshTokenPayload.jti,
                userId: refreshTokenPayload.userId,
                sessionId: refreshTokenPayload.sessionId,
                userAgent: refreshTokenPayload.userAgent,
                ipAddress: refreshTokenPayload.ipAddress,
            }),
            'hashed-token',
            expect.any(Date),
            undefined,
        );
    });

    it('sets expiresAt to session expiresAt if session expires before refresh token TTL', async () => {
        jest.spyOn(Argon2HashUtil, 'hash').mockResolvedValue('hashed-token');

        const earlierSessionExpiresAt = new Date(Date.now() + 1000 * 60 * 60 * 24 * 3); // 3 days (TTL is 7 days)
        sessionRepository.findById.mockResolvedValue({
            id: 'session-id',
            expiresAt: earlierSessionExpiresAt,
        } as unknown as SessionSelect);

        await service.saveRefreshToken(refreshTokenPayload, 'raw-token');

        expect(repository.createRefreshToken).toHaveBeenCalledWith(
            expect.any(Object),
            'hashed-token',
            earlierSessionExpiresAt,
            undefined,
        );
    });

    it('sets expiresAt to refresh token TTL if session expires after it', async () => {
        jest.spyOn(Argon2HashUtil, 'hash').mockResolvedValue('hashed-token');

        const laterSessionExpiresAt = new Date(Date.now() + 1000 * 60 * 60 * 24 * 10); // 10 days (TTL is 7 days)
        sessionRepository.findById.mockResolvedValue({
            id: 'session-id',
            expiresAt: laterSessionExpiresAt,
        } as unknown as SessionSelect);

        await service.saveRefreshToken(refreshTokenPayload, 'raw-token');

        const callArgs = repository.createRefreshToken.mock.calls[0];
        const tokenExpiresAt = callArgs[2];

        // Should be around 7 days from now (default TTL)
        const expectedDiff = 7 * 24 * 60 * 60 * 1000;
        const diff = Math.abs(tokenExpiresAt.getTime() - (Date.now() + expectedDiff));
        expect(diff).toBeLessThan(5000); // 5s delta allowance
    });
});
