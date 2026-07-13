import { ConfigService } from '@nestjs/config';
import { UnauthorizedException } from '@nestjs/common';
import { JwtAdminAuthStrategy } from '@/module-auth/strategies/jwt-admin-auth.strategy';
import { COOKIE_NAMES, UserRoles } from '@/module-auth/enums/auth.enums';
import { Request } from 'express';
import { AccessTokenPayload } from '@/module-auth-token/interfaces/auth-token.interfaces';
import { TokenType } from '@/module-auth-token/enums/auth-token.enums';

describe('JwtAdminAuthStrategy', () => {
    let strategy: JwtAdminAuthStrategy;
    let configService: ConfigService;

    beforeEach(() => {
        configService = {
            getOrThrow: jest.fn((key: string) => {
                const config: Record<string, string> = {
                    JWT_SECRET: 'test-secret',
                    JWT_ACCESS_TOKEN_TTL: '15m',
                };
                return config[key];
            }),
        } as unknown as ConfigService;

        strategy = new JwtAdminAuthStrategy(configService);
    });

    describe('cookie extractor', () => {
        it('extracts token from adminAccessToken cookie', () => {
            const mockRequest = {
                cookies: {
                    [COOKIE_NAMES.ADMIN_ACCESS_TOKEN]: 'mock-admin-token',
                },
            } as unknown as Request;

            const extractor = (
                strategy as unknown as { _jwtFromRequest: (req: Request) => string | null }
            )._jwtFromRequest;
            expect(extractor(mockRequest)).toBe('mock-admin-token');
        });

        it('returns null if adminAccessToken is missing', () => {
            const mockRequest = {
                cookies: {
                    [COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN]: 'mock-customer-token',
                },
            } as unknown as Request;

            const extractor = (
                strategy as unknown as { _jwtFromRequest: (req: Request) => string | null }
            )._jwtFromRequest;
            expect(extractor(mockRequest)).toBeNull();
        });

        it('returns null if cookies are undefined', () => {
            const mockRequest = {} as unknown as Request;

            const extractor = (
                strategy as unknown as { _jwtFromRequest: (req: Request) => string | null }
            )._jwtFromRequest;
            expect(extractor(mockRequest)).toBeNull();
        });
    });

    describe('validate', () => {
        it('accepts admin payload with admin role', () => {
            const payload: AccessTokenPayload = {
                sub: 'user-1',
                userId: 'user-1',
                type: TokenType.ACCESS,
                ipAddress: '127.0.0.1',
                userAgent: 'test',
                email: 'admin@example.com',
                username: 'admin',
                sessionId: 'session-1',
                jti: 'jti-1',
                role: UserRoles.PLATFORM_ADMIN,
                realm: 'admin',
            };

            const result = strategy.validate(payload);
            expect(result).toEqual(payload);
        });

        it('accepts admin payload with super admin role', () => {
            const payload: AccessTokenPayload = {
                sub: 'user-2',
                userId: 'user-2',
                type: TokenType.ACCESS,
                ipAddress: '127.0.0.1',
                userAgent: 'test',
                email: 'super-admin@example.com',
                username: 'super-admin',
                sessionId: 'session-2',
                jti: 'jti-2',
                role: UserRoles.SUPER_ADMIN,
                realm: 'admin',
            };

            const result = strategy.validate(payload);
            expect(result).toEqual(payload);
        });

        it('rejects admin payload with customer user role', () => {
            const payload: AccessTokenPayload = {
                sub: 'user-3',
                userId: 'user-3',
                type: TokenType.ACCESS,
                ipAddress: '127.0.0.1',
                userAgent: 'test',
                email: 'user@example.com',
                username: 'user',
                sessionId: 'session-3',
                jti: 'jti-3',
                role: UserRoles.USER,
                realm: 'admin',
            };

            expect(() => strategy.validate(payload)).toThrow(UnauthorizedException);
        });

        it('rejects payload if realm is customer', () => {
            const payload: AccessTokenPayload = {
                sub: 'user-4',
                userId: 'user-4',
                type: TokenType.ACCESS,
                ipAddress: '127.0.0.1',
                userAgent: 'test',
                email: 'admin@example.com',
                username: 'admin',
                sessionId: 'session-4',
                jti: 'jti-4',
                role: UserRoles.PLATFORM_ADMIN,
                realm: 'customer',
            };

            expect(() => strategy.validate(payload)).toThrow(UnauthorizedException);
        });
    });
});
