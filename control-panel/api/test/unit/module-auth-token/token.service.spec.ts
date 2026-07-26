import { JwtService } from '@nestjs/jwt';
import { ConfigService } from '@nestjs/config';
import { TokenService } from '@/module-auth-token/services/token.service';
import {
    AccessTokenPayload,
    RefreshTokenPayload,
} from '@/module-auth-token/interfaces/auth-token.interfaces';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';

describe('TokenService', () => {
    let service: TokenService;
    let jwtService: {
        sign: jest.MockedFunction<JwtService['sign']>;
        verify: jest.MockedFunction<JwtService['verify']>;
    };
    let configService: Pick<ConfigService, 'get' | 'getOrThrow'>;

    beforeEach(() => {
        jwtService = {
            sign: jest.fn().mockReturnValue('signed'),
            verify: jest.fn((token: string, options?: { secret?: string }) => {
                if (token === 'previous-token' && options?.secret === 'p'.repeat(32)) {
                    return { userId: 'previous-user' };
                }
                if (token === 'previous-token') {
                    throw new Error('invalid signature');
                }
                if (options?.secret === 'j'.repeat(32)) {
                    return { userId: 'user-id' };
                }
                throw new Error('invalid signature');
            }) as jest.MockedFunction<JwtService['verify']>,
        };
        configService = {
            getOrThrow: ((key: string) => {
                if (key === 'JWT_ACCESS_TOKEN_TTL') return '15m';
                if (key === 'JWT_REFRESH_TOKEN_TTL') return '7d';
                if (key === 'JWT_SECRET') return 'j'.repeat(32);
                return undefined;
            }) as ConfigService['getOrThrow'],
            get: ((key: string) =>
                key === 'JWT_PREVIOUS_SECRETS' ? ['p'.repeat(32)] : undefined) as ConfigService['get'],
        };

        service = new TokenService(
            configService as unknown as ConfigService,
            jwtService as unknown as JwtService,
        );
    });

    it('creates access tokens with configured ttl', () => {
        const payload = {
            sub: 'user-id',
            userId: 'user-id',
            sessionId: 'session-id',
            email: 'john@example.com',
            username: 'john',
            role: 'user',
            realm: 'customer',
            ipAddress: '1.1.1.1',
            userAgent: 'ua',
            type: TokenType.ACCESS,
            jti: 'id',
        } satisfies AccessTokenPayload;

        const token = service.createAccessToken(payload);

        expect(token).toBe('signed');
        expect(jwtService.sign).toHaveBeenCalledWith(payload, { expiresIn: '15m' });
    });

    it('creates refresh tokens with refresh ttl and refresh type', () => {
        const payload = {
            sub: 'user-id',
            userId: 'user-id',
            sessionId: 'session-id',
            ipAddress: '1.1.1.1',
            userAgent: 'ua',
            type: TokenType.REFRESH,
            jti: 'id',
            tokenId: 'token-id',
        } satisfies RefreshTokenPayload;

        const token = service.createRefreshToken(payload);

        expect(token).toBe('signed');
        expect(jwtService.sign).toHaveBeenCalledWith(
            { ...payload, type: 'refresh' },
            { expiresIn: '7d' },
        );
    });

    it('verifies incoming jwt tokens', () => {
        const decoded = service.verifyToken('token');

        expect(decoded).toEqual({ userId: 'user-id' });
        expect(jwtService.verify).toHaveBeenCalledWith('token', { secret: 'j'.repeat(32) });
    });

    it('verifies tokens signed with a previous jwt secret', () => {
        const decoded = service.verifyToken('previous-token');

        expect(decoded).toEqual({ userId: 'previous-user' });
        expect(jwtService.verify).toHaveBeenCalledWith('previous-token', {
            secret: 'j'.repeat(32),
        });
        expect(jwtService.verify).toHaveBeenCalledWith('previous-token', {
            secret: 'p'.repeat(32),
        });
    });
});
