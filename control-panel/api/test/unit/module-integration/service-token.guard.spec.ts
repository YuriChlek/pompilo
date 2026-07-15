import { ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { ServiceTokenGuard } from '@/module-integration/guards/service-token.guard';

describe('ServiceTokenGuard', () => {
    const buildContext = (headers: Record<string, string | undefined>): ExecutionContext =>
        ({
            switchToHttp: () => ({
                getRequest: () => ({
                    header: (name: string) => headers[name.toLowerCase()],
                }),
            }),
        }) as unknown as ExecutionContext;

    it('allows matching bearer service tokens', () => {
        const guard = new ServiceTokenGuard({
            get: jest.fn().mockReturnValue('service-secret-token'),
        } as unknown as ConfigService);

        expect(
            guard.canActivate(
                buildContext({
                    authorization: 'Bearer service-secret-token',
                }),
            ),
        ).toBe(true);
    });

    it('rejects invalid service tokens', () => {
        const guard = new ServiceTokenGuard({
            get: jest.fn().mockReturnValue('service-secret-token'),
        } as unknown as ConfigService);

        expect(() =>
            guard.canActivate(
                buildContext({
                    authorization: 'Bearer wrong-token',
                }),
            ),
        ).toThrow(UnauthorizedException);
    });

    it('rejects requests when no service secret is configured', () => {
        const guard = new ServiceTokenGuard({
            get: jest.fn().mockReturnValue(undefined),
        } as unknown as ConfigService);

        expect(() => guard.canActivate(buildContext({}))).toThrow(UnauthorizedException);
    });
});
