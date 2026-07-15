import { AdminMailActionRateLimitGuard } from '@/module-mail/guards/admin-mail-action-rate-limit.guard';
import { RedisService } from '@/common/redis/redis.service';
import { Reflector } from '@nestjs/core';
import { ExecutionContext } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ADMIN_MAIL_ACTION_RATE_LIMIT_MESSAGE } from '@/module-mail/constants/admin-mail-action-rate-limit.constants';
import { createHash } from 'crypto';

describe('AdminMailActionRateLimitGuard', () => {
    let redis: {
        eval: jest.Mock;
        pttl: jest.Mock;
    };
    let reflector: {
        getAllAndOverride: jest.Mock;
    };
    let configService: {
        get: jest.Mock;
    };
    let guard: AdminMailActionRateLimitGuard;

    const response = {
        setHeader: jest.fn(),
    };

    const buildContext = (
        user: { userId?: string } | undefined,
        headers: Record<string, string> = {},
    ): ExecutionContext =>
        ({
            getHandler: jest.fn(),
            getClass: jest.fn(),
            switchToHttp: () => ({
                getRequest: () => ({
                    user,
                    ip: '203.0.113.10',
                    socket: {},
                    headers,
                }),
                getResponse: () => response,
            }),
        }) as unknown as ExecutionContext;

    beforeEach(() => {
        redis = {
            eval: jest.fn().mockResolvedValue(1),
            pttl: jest.fn().mockResolvedValue(30000),
        };
        reflector = {
            getAllAndOverride: jest.fn().mockReturnValue({ action: 'test_email' }),
        };
        configService = {
            get: jest.fn().mockReturnValue('production'),
        };
        response.setHeader.mockReset();

        guard = new AdminMailActionRateLimitGuard(
            reflector as unknown as Reflector,
            { getClient: () => redis } as unknown as RedisService,
            configService as unknown as ConfigService,
        );
    });

    it('counts admin verify attempts in Redis before handler execution in production', async () => {
        await expect(guard.canActivate(buildContext({ userId: 'admin-1' }))).resolves.toBe(true);

        expect(redis.eval).toHaveBeenCalledTimes(1);
        const calls = redis.eval.mock.calls as unknown[][];
        const key = calls[0][2] as string;
        expect(key).toContain('rate-limit:admin-mail:test_email:');
        expect(key).not.toContain('admin-1');
    });

    it('ignores spoofed forwarding headers when falling back to IP identity', async () => {
        await expect(
            guard.canActivate(buildContext(undefined, { 'x-forwarded-for': '198.51.100.77' })),
        ).resolves.toBe(true);

        const calls = redis.eval.mock.calls as unknown[][];
        const key = calls[0][2] as string;
        const spoofedIdentifierHash = createHash('sha256').update('ip:198.51.100.77').digest('hex');

        expect(key).toContain('rate-limit:admin-mail:test_email:');
        expect(key).not.toContain(spoofedIdentifierHash);
    });

    it('rejects limited admin mail actions with safe response in production', async () => {
        redis.eval.mockResolvedValue(4);

        await expect(guard.canActivate(buildContext({ userId: 'admin-1' }))).rejects.toMatchObject({
            response: {
                statusCode: 429,
                message: ADMIN_MAIL_ACTION_RATE_LIMIT_MESSAGE,
            },
        });

        expect(response.setHeader).toHaveBeenCalledWith('Retry-After', '30');
    });

    it('bypasses rate limiting in non-production environments', async () => {
        configService.get.mockReturnValue('development');
        redis.eval.mockResolvedValue(100);

        await expect(guard.canActivate(buildContext({ userId: 'admin-1' }))).resolves.toBe(true);
        expect(redis.eval).not.toHaveBeenCalled();
    });
});
