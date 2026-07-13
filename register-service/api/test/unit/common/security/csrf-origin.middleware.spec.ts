import { createCsrfOriginMiddleware } from '@/common/security/csrf-origin.middleware';
import { COOKIE_NAMES } from '@/module-auth/enums/auth.enums';
import type { NextFunction, Request, Response } from 'express';

describe('createCsrfOriginMiddleware', () => {
    function createResponse() {
        return {
            status: jest.fn().mockReturnThis(),
            json: jest.fn(),
        } as unknown as Response & { status: jest.Mock; json: jest.Mock };
    }

    function createNext(): jest.MockedFunction<NextFunction> {
        return jest.fn() as jest.MockedFunction<NextFunction>;
    }

    it('allows unsafe authenticated requests from configured origins', () => {
        const middleware = createCsrfOriginMiddleware({
            enabled: true,
            allowedOrigins: ['https://app.example.com'],
        });
        const response = createResponse();
        const next = createNext();

        middleware(
            {
                method: 'POST',
                headers: { origin: 'https://app.example.com/settings' },
                cookies: { [COOKIE_NAMES.CUSTOMER_ACCESS_TOKEN]: 'token' },
            } as unknown as Request,
            response,
            next,
        );

        expect(next).toHaveBeenCalledTimes(1);
        expect(response.status).not.toHaveBeenCalled();
    });

    it('rejects unsafe authenticated requests from untrusted origins', () => {
        const middleware = createCsrfOriginMiddleware({
            enabled: true,
            allowedOrigins: ['https://app.example.com'],
        });
        const response = createResponse();
        const next = createNext();

        middleware(
            {
                method: 'DELETE',
                headers: { origin: 'https://evil.example.com' },
                cookies: { [COOKIE_NAMES.ADMIN_ACCESS_TOKEN]: 'token' },
            } as unknown as Request,
            response,
            next,
        );

        expect(next).not.toHaveBeenCalled();
        expect(response.status).toHaveBeenCalledWith(403);
        expect(response.json).toHaveBeenCalledWith(
            expect.objectContaining({ message: 'CSRF origin check failed' }),
        );
    });

    it('skips requests without auth cookies', () => {
        const middleware = createCsrfOriginMiddleware({
            enabled: true,
            allowedOrigins: ['https://app.example.com'],
        });
        const response = createResponse();
        const next = createNext();

        middleware(
            {
                method: 'POST',
                headers: { origin: 'https://evil.example.com' },
                cookies: {},
            } as unknown as Request,
            response,
            next,
        );

        expect(next).toHaveBeenCalledTimes(1);
        expect(response.status).not.toHaveBeenCalled();
    });
});
