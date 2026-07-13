import { Reflector } from '@nestjs/core';
import { ExecutionContext, ServiceUnavailableException } from '@nestjs/common';
import { JwtAuthGuard } from '@/module-auth/guards/jwt-auth.guard';
import { RedisTokenService } from '@/module-auth-token/services/redis-token.service';

describe('JwtAuthGuard', () => {
    let guard: JwtAuthGuard;
    let redisTokenService: {
        isTokenRevoked: jest.Mock;
        isSessionRevoked: jest.Mock;
    };
    let reflector: {
        getAllAndOverride: jest.Mock;
    };

    beforeEach(() => {
        redisTokenService = {
            isTokenRevoked: jest.fn(),
            isSessionRevoked: jest.fn(),
        };
        reflector = {
            getAllAndOverride: jest.fn().mockReturnValue(false),
        };
        guard = new JwtAuthGuard(
            redisTokenService as unknown as RedisTokenService,
            reflector as unknown as Reflector,
        );
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    const createMockContext = (request = {}) =>
        ({
            switchToHttp: () => ({
                getRequest: () => request,
            }),
            getHandler: () => ({}),
            getClass: () => ({}),
        }) as unknown as ExecutionContext;

    it('returns true immediately if endpoint is public', async () => {
        reflector.getAllAndOverride.mockReturnValue(true);

        const mockContext = createMockContext();

        const result = await guard.canActivate(mockContext);
        expect(result).toBe(true);
        expect(reflector.getAllAndOverride).toHaveBeenCalled();
    });

    it('returns false if passport authentication fails', async () => {
        reflector.getAllAndOverride.mockReturnValue(false);
        jest.spyOn(Object.getPrototypeOf(JwtAuthGuard.prototype), 'canActivate').mockResolvedValue(
            false,
        );

        const mockContext = createMockContext();

        const result = await guard.canActivate(mockContext);
        expect(result).toBe(false);
    });

    it('returns true if token and session are valid and not revoked', async () => {
        jest.spyOn(Object.getPrototypeOf(JwtAuthGuard.prototype), 'canActivate').mockResolvedValue(
            true,
        );

        const request = {
            user: { userId: 'user1', sessionId: 'session-1', jti: 'valid-jti' },
        };
        const mockContext = createMockContext(request);

        redisTokenService.isTokenRevoked.mockResolvedValue(false);
        redisTokenService.isSessionRevoked.mockResolvedValue(false);

        const result = await guard.canActivate(mockContext);
        expect(result).toBe(true);
        expect(redisTokenService.isTokenRevoked).toHaveBeenCalledWith('valid-jti');
        expect(redisTokenService.isSessionRevoked).toHaveBeenCalledWith('session-1');
    });

    it('returns false if token is revoked in Redis', async () => {
        jest.spyOn(Object.getPrototypeOf(JwtAuthGuard.prototype), 'canActivate').mockResolvedValue(
            true,
        );

        const request = {
            user: { userId: 'user1', sessionId: 'session-1', jti: 'revoked-jti' },
        };
        const mockContext = createMockContext(request);

        redisTokenService.isTokenRevoked.mockResolvedValue(true);
        redisTokenService.isSessionRevoked.mockResolvedValue(false);

        const result = await guard.canActivate(mockContext);
        expect(result).toBe(false);
        expect(redisTokenService.isTokenRevoked).toHaveBeenCalledWith('revoked-jti');
    });

    it('returns false if session is revoked in Redis', async () => {
        jest.spyOn(Object.getPrototypeOf(JwtAuthGuard.prototype), 'canActivate').mockResolvedValue(
            true,
        );

        const request = {
            user: { userId: 'user1', sessionId: 'session-1', jti: 'valid-jti' },
        };
        const mockContext = createMockContext(request);

        redisTokenService.isTokenRevoked.mockResolvedValue(false);
        redisTokenService.isSessionRevoked.mockResolvedValue(true);

        const result = await guard.canActivate(mockContext);
        expect(result).toBe(false);
        expect(redisTokenService.isSessionRevoked).toHaveBeenCalledWith('session-1');
    });

    it('returns false if authenticated token has no session id', async () => {
        jest.spyOn(Object.getPrototypeOf(JwtAuthGuard.prototype), 'canActivate').mockResolvedValue(
            true,
        );

        const request = {
            user: { userId: 'user1', jti: 'valid-jti' },
        };
        const mockContext = createMockContext(request);

        const result = await guard.canActivate(mockContext);
        expect(result).toBe(false);
        expect(redisTokenService.isTokenRevoked).not.toHaveBeenCalled();
        expect(redisTokenService.isSessionRevoked).not.toHaveBeenCalled();
    });

    it('throws ServiceUnavailableException (503) if isTokenRevoked fails due to Redis error', async () => {
        jest.spyOn(Object.getPrototypeOf(JwtAuthGuard.prototype), 'canActivate').mockResolvedValue(
            true,
        );

        const request = {
            user: { userId: 'user1', sessionId: 'session-1', jti: 'valid-jti' },
        };
        const mockContext = createMockContext(request);

        redisTokenService.isTokenRevoked.mockRejectedValue(new Error('Redis connection lost'));

        await expect(guard.canActivate(mockContext)).rejects.toThrow(ServiceUnavailableException);
    });

    it('throws ServiceUnavailableException (503) if isSessionRevoked fails due to Redis error', async () => {
        jest.spyOn(Object.getPrototypeOf(JwtAuthGuard.prototype), 'canActivate').mockResolvedValue(
            true,
        );

        const request = {
            user: { userId: 'user1', sessionId: 'session-1', jti: 'valid-jti' },
        };
        const mockContext = createMockContext(request);

        redisTokenService.isTokenRevoked.mockResolvedValue(false);
        redisTokenService.isSessionRevoked.mockRejectedValue(new Error('Redis connection lost'));

        await expect(guard.canActivate(mockContext)).rejects.toThrow(ServiceUnavailableException);
    });
});
