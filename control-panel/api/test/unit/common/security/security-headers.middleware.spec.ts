import { createSecurityHeadersMiddleware } from '@/common/security/security-headers.middleware';
import type { NextFunction, Request, Response } from 'express';

describe('createSecurityHeadersMiddleware', () => {
    it('sets baseline browser security headers', () => {
        const headers = new Map<string, string | number | readonly string[]>();
        const middleware = createSecurityHeadersMiddleware('production');
        const next = jest.fn() as jest.MockedFunction<NextFunction>;

        middleware(
            {} as Request,
            {
                setHeader: (key, value) => {
                    headers.set(key, value);
                    return undefined as unknown as Response;
                },
            } as unknown as Response,
            next,
        );

        expect(headers.get('X-Content-Type-Options')).toBe('nosniff');
        expect(headers.get('X-Frame-Options')).toBe('DENY');
        expect(headers.get('Strict-Transport-Security')).toBe(
            'max-age=31536000; includeSubDomains',
        );
        expect(next).toHaveBeenCalledTimes(1);
    });
});
