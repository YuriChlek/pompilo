import { BadRequestException } from '@nestjs/common';
import type { ExecutionContext, CallHandler } from '@nestjs/common';
import type { Response, Request } from 'express';
import { Observable, of, throwError, lastValueFrom } from 'rxjs';
import { ResponseInterceptor } from '@/common/interceptors/response.interceptor';
import { ResponseData } from '@/common/dto/response.dto';
import { Reflector } from '@nestjs/core';

const createExecutionContext = (): ExecutionContext =>
    ({
        getHandler: () => createExecutionContext,
        getClass: () => ResponseInterceptor,
        switchToHttp: () => ({
            getResponse: () =>
                ({
                    statusCode: 200,
                }) as Response,
            getRequest: () =>
                ({
                    url: '/test',
                }) as Request,
        }),
    }) as ExecutionContext;

describe('ResponseInterceptor', () => {
    let interceptor: ResponseInterceptor<unknown>;
    let reflector: { getAllAndOverride: jest.Mock };
    let context: ExecutionContext;

    beforeEach(() => {
        reflector = {
            getAllAndOverride: jest.fn().mockReturnValue(false),
        };
        interceptor = new ResponseInterceptor(reflector as unknown as Reflector);
        context = createExecutionContext();
    });

    it('wraps successful responses with metadata', async () => {
        const handler: CallHandler = {
            handle: () => of({ message: 'ok' }),
        };

        const result = await lastValueFrom(
            interceptor.intercept(context, handler) as Observable<ResponseData<unknown>>,
        );

        expect(result).toMatchObject({
            success: true,
            statusCode: 200,
            data: { message: 'ok' },
        });
        expect(result.timestamp).toBeDefined();
    });

    it('returns raw response data when the handler opts out of the response envelope', async () => {
        reflector.getAllAndOverride.mockReturnValue(true);
        const handler: CallHandler = {
            handle: () => of('mail_outbox_lag_seconds 0\n'),
        };

        const result = await lastValueFrom(interceptor.intercept(context, handler) as Observable<string>);

        expect(result).toBe('mail_outbox_lag_seconds 0\n');
    });

    it('propagates thrown exceptions without mutating them', async () => {
        const handler: CallHandler = {
            handle: () => throwError(() => new BadRequestException('Invalid')),
        };

        await expect(
            lastValueFrom(
                interceptor.intercept(context, handler) as Observable<ResponseData<unknown>>,
            ),
        ).rejects.toBeInstanceOf(BadRequestException);
    });
});
