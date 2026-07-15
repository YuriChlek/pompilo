import type { NextFunction, Request, Response } from 'express';

export function createSecurityHeadersMiddleware(nodeEnv: string) {
    const isProduction = nodeEnv === 'production';

    return (_request: Request, response: Response, next: NextFunction): void => {
        response.setHeader('X-Content-Type-Options', 'nosniff');
        response.setHeader('X-Frame-Options', 'DENY');
        response.setHeader('Referrer-Policy', 'no-referrer');
        response.setHeader('Permissions-Policy', 'camera=(), microphone=(), geolocation=()');
        response.setHeader('Cross-Origin-Resource-Policy', 'same-site');
        response.setHeader(
            'Content-Security-Policy',
            "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; object-src 'none'",
        );

        if (isProduction) {
            response.setHeader('Strict-Transport-Security', 'max-age=31536000; includeSubDomains');
        }

        next();
    };
}
