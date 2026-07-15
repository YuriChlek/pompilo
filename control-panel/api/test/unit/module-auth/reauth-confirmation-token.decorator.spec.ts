import { ExecutionContext } from '@nestjs/common';
import { ROUTE_ARGS_METADATA } from '@nestjs/common/constants';
import { ReauthConfirmationToken } from '@/module-auth/decorators/reauth-confirmation-token.decorator';

describe('ReauthConfirmationToken Decorator', () => {
    it('should extract token from x-reauth-confirmation header', () => {
        class TestController {
            test(@ReauthConfirmationToken() token: string) {
                return token;
            }
        }

        const metadata = Reflect.getMetadata(ROUTE_ARGS_METADATA, TestController, 'test') as Record<
            string,
            { factory: (data: unknown, ctx: ExecutionContext) => unknown }
        >;
        const key = Object.keys(metadata)[0];
        const factory = metadata[key].factory;

        const mockRequest = {
            headers: {
                'x-reauth-confirmation': 'test-token',
            },
        };

        const mockExecutionContext = {
            switchToHttp: () => ({
                getRequest: () => mockRequest,
            }),
        } as unknown as ExecutionContext;

        const result = factory(null, mockExecutionContext);
        expect(result).toBe('test-token');
    });

    it('should handle array token and return the first element', () => {
        class TestController {
            test(@ReauthConfirmationToken() token: string) {
                return token;
            }
        }

        const metadata = Reflect.getMetadata(ROUTE_ARGS_METADATA, TestController, 'test') as Record<
            string,
            { factory: (data: unknown, ctx: ExecutionContext) => unknown }
        >;
        const key = Object.keys(metadata)[0];
        const factory = metadata[key].factory;

        const mockRequest = {
            headers: {
                'x-reauth-confirmation': ['token-1', 'token-2'],
            },
        };

        const mockExecutionContext = {
            switchToHttp: () => ({
                getRequest: () => mockRequest,
            }),
        } as unknown as ExecutionContext;

        const result = factory(null, mockExecutionContext);
        expect(result).toBe('token-1');
    });

    it('should return undefined if header is missing', () => {
        class TestController {
            test(@ReauthConfirmationToken() token: string) {
                return token;
            }
        }

        const metadata = Reflect.getMetadata(ROUTE_ARGS_METADATA, TestController, 'test') as Record<
            string,
            { factory: (data: unknown, ctx: ExecutionContext) => unknown }
        >;
        const key = Object.keys(metadata)[0];
        const factory = metadata[key].factory;

        const mockRequest = {
            headers: {},
        };

        const mockExecutionContext = {
            switchToHttp: () => ({
                getRequest: () => mockRequest,
            }),
        } as unknown as ExecutionContext;

        const result = factory(null, mockExecutionContext);
        expect(result).toBeUndefined();
    });
});
