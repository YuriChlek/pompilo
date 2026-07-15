import { InternalServerErrorException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import type { Request, Response } from 'express';
import { AuthSessionService } from '@/module-auth/services/auth-session.service';
import { AuthTokenService } from '@/module-auth-token/services/auth-token.service';
import { COOKIE_NAMES, UserRoles } from '@/module-auth/enums/auth.enums';
import type { UserJwtPayload } from '@/module-user/interfaces/user.interfaces';

const createConfigService = (): Pick<ConfigService, 'get' | 'getOrThrow'> => ({
    get: jest.fn((key: string) => {
        const config: Record<string, string> = {
            NODE_ENV: 'test',
        };

        return config[key];
    }),
    getOrThrow: jest.fn((key: string) => {
        const config: Record<string, string> = {
            COOKIE_DOMAIN: 'localhost',
            JWT_ACCESS_TOKEN_TTL: '30m',
            JWT_REFRESH_TOKEN_TTL: '7d',
        };

        return config[key];
    }),
});

const createResponse = (): Response =>
    ({
        cookie: jest.fn(),
    }) as unknown as Response;

const getCookieMock = (response: Response): jest.Mock =>
    (response as unknown as { cookie: jest.Mock }).cookie;

const createPayload = (role: UserRoles): UserJwtPayload => ({
    id: 'user-id',
    name: 'John',
    email: 'john@example.com',
    role,
    ipAddress: '127.0.0.1',
    sessionId: 'session-id',
    userAgent: '',
});

describe('AuthSessionService', () => {
    let service: AuthSessionService;
    let authTokenService: {
        createAccessToken: jest.MockedFunction<AuthTokenService['createAccessToken']>;
        createRefreshToken: jest.MockedFunction<AuthTokenService['createRefreshToken']>;
        createInitialRefreshToken: jest.MockedFunction<
            AuthTokenService['createInitialRefreshToken']
        >;
    };

    beforeEach(() => {
        authTokenService = {
            createAccessToken: jest.fn().mockReturnValue('access-token'),
            createRefreshToken: jest.fn().mockResolvedValue('refresh-token'),
            createInitialRefreshToken: jest.fn().mockResolvedValue('refresh-token'),
        };

        service = new AuthSessionService(
            authTokenService as unknown as AuthTokenService,
            createConfigService() as ConfigService,
        );
    });

    describe('getUserMetaData', () => {
        it('uses request.ip as canonical IP when populated', () => {
            const request = {
                ip: '203.0.113.50',
                headers: {
                    'user-agent': 'Mozilla/5.0',
                    'x-forwarded-for': '198.51.100.1, 203.0.113.50',
                },
                socket: {
                    remoteAddress: '127.0.0.1',
                },
            } as unknown as Request;

            expect(service.getUserMetaData(request)).toEqual({
                ipAddress: '203.0.113.50',
                userAgent: 'Mozilla/5.0',
            });
        });

        it('ignores x-forwarded-for header and falls back to socket.remoteAddress if request.ip is not populated', () => {
            const request = {
                headers: {
                    'x-forwarded-for': '198.51.100.1',
                },
                socket: {
                    remoteAddress: '127.0.0.1',
                },
            } as unknown as Request;

            expect(service.getUserMetaData(request)).toEqual({
                ipAddress: '127.0.0.1',
                userAgent: '',
            });
        });

        it('uses empty user agent fallback when request header is missing', () => {
            const request = {
                headers: {},
                socket: {
                    remoteAddress: '127.0.0.1',
                },
            } as unknown as Request;

            expect(service.getUserMetaData(request)).toEqual({
                ipAddress: '127.0.0.1',
                userAgent: '',
            });
        });
    });

    it('does not set partial cookies when refresh token creation fails', async () => {
        const response = createResponse();

        authTokenService.createInitialRefreshToken.mockRejectedValue(
            new Error('token store failed'),
        );

        await expect(
            service.issueTokens(response, {
                id: 'user-id',
                name: 'John',
                email: 'john@example.com',
                role: UserRoles.USER,
                ipAddress: '127.0.0.1',
                userAgent: '',
            }),
        ).rejects.toBeInstanceOf(InternalServerErrorException);

        expect((response.cookie as jest.Mock).mock.calls).toHaveLength(0);
    });

    it('sets admin access and refresh cookies with the proxy-compatible / path invariant', async () => {
        const response = createResponse();
        const cookieMock = getCookieMock(response);

        await service.issueTokens(response, createPayload(UserRoles.PLATFORM_ADMIN));

        expect(cookieMock).toHaveBeenNthCalledWith(
            1,
            COOKIE_NAMES.ADMIN_ACCESS_TOKEN,
            'access-token',
            expect.objectContaining({
                path: '/',
                httpOnly: true,
                sameSite: 'lax',
                domain: 'localhost',
                secure: false,
            }),
        );
        expect(cookieMock).toHaveBeenNthCalledWith(
            2,
            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
            'refresh-token',
            expect.objectContaining({
                path: '/',
                httpOnly: true,
                sameSite: 'lax',
                domain: 'localhost',
                secure: false,
            }),
        );
    });

    it('uses the same proxy-compatible / cookie path and admin cookie names for super admin sessions', async () => {
        const response = createResponse();
        const cookieMock = getCookieMock(response);

        await service.issueTokens(response, createPayload(UserRoles.SUPER_ADMIN));

        expect(cookieMock).toHaveBeenNthCalledWith(
            1,
            COOKIE_NAMES.ADMIN_ACCESS_TOKEN,
            'access-token',
            expect.objectContaining({ path: '/' }),
        );
        expect(cookieMock).toHaveBeenNthCalledWith(
            2,
            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
            'refresh-token',
            expect.objectContaining({ path: '/' }),
        );
    });

    it('uses admin cookies for the neutral platform admin role', async () => {
        const response = createResponse();
        const cookieMock = getCookieMock(response);

        await service.issueTokens(response, createPayload(UserRoles.PLATFORM_ADMIN));

        expect(cookieMock).toHaveBeenNthCalledWith(
            1,
            COOKIE_NAMES.ADMIN_ACCESS_TOKEN,
            'access-token',
            expect.objectContaining({ path: '/' }),
        );
        expect(cookieMock).toHaveBeenNthCalledWith(
            2,
            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
            'refresh-token',
            expect.objectContaining({ path: '/' }),
        );
    });

    it('does not issue a new refresh cookie when refreshing an admin access token', async () => {
        const response = createResponse();
        const cookieMock = getCookieMock(response);

        await service.issueTokens(response, createPayload(UserRoles.PLATFORM_ADMIN), false);

        expect(authTokenService.createRefreshToken).not.toHaveBeenCalled();
        expect(cookieMock).toHaveBeenCalledTimes(1);
        expect(cookieMock).toHaveBeenCalledWith(
            COOKIE_NAMES.ADMIN_ACCESS_TOKEN,
            'access-token',
            expect.objectContaining({ path: '/' }),
        );
    });

    it('requires an existing session id when refreshing an access token', async () => {
        const response = createResponse();

        await expect(
            service.issueTokens(
                response,
                {
                    id: 'user-id',
                    name: 'John',
                    email: 'john@example.com',
                    role: UserRoles.PLATFORM_ADMIN,
                    ipAddress: '127.0.0.1',
                    userAgent: '',
                },
                false,
            ),
        ).rejects.toBeInstanceOf(InternalServerErrorException);
    });

    it('clears admin cookies with the same / path used to set them', () => {
        const response = createResponse();
        const cookieMock = getCookieMock(response);

        service.clearTokens(response, UserRoles.PLATFORM_ADMIN);

        expect(cookieMock).toHaveBeenNthCalledWith(
            1,
            COOKIE_NAMES.ADMIN_ACCESS_TOKEN,
            '',
            expect.objectContaining({
                path: '/',
                expires: new Date(0),
            }),
        );
        expect(cookieMock).toHaveBeenNthCalledWith(
            2,
            COOKIE_NAMES.ADMIN_REFRESH_TOKEN,
            '',
            expect.objectContaining({
                path: '/',
                expires: new Date(0),
            }),
        );
    });

    it('sets customer access and refresh cookies with path / for user role', async () => {
        const response = createResponse();
        const cookieMock = getCookieMock(response);

        await service.issueTokens(response, createPayload(UserRoles.USER));

        expect(cookieMock).toHaveBeenNthCalledWith(
            1,
            COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN,
            'access-token',
            expect.objectContaining({
                path: '/',
                httpOnly: true,
                sameSite: 'lax',
                domain: 'localhost',
            }),
        );
        expect(cookieMock).toHaveBeenNthCalledWith(
            2,
            COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN,
            'refresh-token',
            expect.objectContaining({
                path: '/',
                httpOnly: true,
                sameSite: 'lax',
                domain: 'localhost',
            }),
        );
    });

});
