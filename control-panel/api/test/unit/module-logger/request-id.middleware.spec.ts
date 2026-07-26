import type { NextFunction, Request, Response } from 'express';
import { createRequestIdMiddleware } from '@/module-logger/middleware/request-id.middleware';
import { getRequestLogContext } from '@/module-logger/utils/request-log-context.util';

describe('createRequestIdMiddleware', () => {
    function createResponseMock() {
        return {
            setHeader: jest.fn(),
        } as unknown as Response & {
            setHeader: jest.Mock;
        };
    }

    it('preserves an incoming request id and exposes it through async context', () => {
        const middleware = createRequestIdMiddleware({ requestIdHeader: 'x-request-id' });
        const request = {
            headers: {
                'x-request-id': 'req-incoming',
            },
        } as unknown as Request;
        const response = createResponseMock();
        const next: NextFunction = jest.fn(() => {
            expect(getRequestLogContext()).toEqual({ requestId: 'req-incoming' });
        });

        middleware(request, response, next);

        expect(response.setHeader).toHaveBeenCalledWith('x-request-id', 'req-incoming');
        expect(next).toHaveBeenCalledTimes(1);
    });

    it('generates a request id when the incoming header is absent', () => {
        const middleware = createRequestIdMiddleware({ requestIdHeader: 'x-request-id' });
        const request = {
            headers: {},
        } as unknown as Request;
        const response = createResponseMock();
        const next: NextFunction = jest.fn(() => {
            expect(getRequestLogContext()?.requestId).toMatch(
                /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
            );
        });

        middleware(request, response, next);

        expect(response.setHeader).toHaveBeenCalledWith(
            'x-request-id',
            expect.stringMatching(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/),
        );
        expect(next).toHaveBeenCalledTimes(1);
    });

    it('ignores invalid incoming request ids before setting the response header', () => {
        const middleware = createRequestIdMiddleware({ requestIdHeader: 'x-request-id' });
        const request = {
            headers: {
                'x-request-id': 'bad\nrequest-id',
            },
        } as unknown as Request;
        const response = createResponseMock();
        const next: NextFunction = jest.fn();

        middleware(request, response, next);

        expect(response.setHeader).toHaveBeenCalledWith(
            'x-request-id',
            expect.not.stringContaining('\n'),
        );
        expect(next).toHaveBeenCalledTimes(1);
    });

    it('supports a configured request id header name', () => {
        const middleware = createRequestIdMiddleware({ requestIdHeader: 'x-correlation-id' });
        const request = {
            headers: {
                'x-correlation-id': 'correlation-id',
            },
        } as unknown as Request;
        const response = createResponseMock();
        const next: NextFunction = jest.fn(() => {
            expect(getRequestLogContext()).toEqual({ requestId: 'correlation-id' });
        });

        middleware(request, response, next);

        expect(response.setHeader).toHaveBeenCalledWith('x-correlation-id', 'correlation-id');
        expect(next).toHaveBeenCalledTimes(1);
    });
});
