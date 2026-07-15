import { describe, it, expect } from 'vitest';
import { decodeJwt, isJwtExpired } from '@/features/module-auth/lib/jwt';

describe('JWT Utility', () => {
    const createToken = (payload: Record<string, unknown>) => {
        const header = btoa(JSON.stringify({ alg: 'HS256', typ: 'JWT' }));
        const payloadStr = btoa(JSON.stringify(payload))
            .replace(/=/g, '')
            .replace(/\+/g, '-')
            .replace(/\//g, '_');
        return `${header}.${payloadStr}.signature`;
    };

    it('should decode a valid JWT token', () => {
        const payload = { id: '123', name: 'Test', email: 'test@example.com', role: 'user' };
        const token = createToken(payload);
        const decoded = decodeJwt(token);
        expect(decoded).toMatchObject(payload);
    });

    it('should return null for invalid token format', () => {
        expect(decodeJwt('invalid-token')).toBeNull();
        expect(decodeJwt('a.b')).toBeNull();
    });

    it('should identify expired tokens', () => {
        const expiredToken = createToken({ exp: Math.floor(Date.now() / 1000) - 10 });
        expect(isJwtExpired(expiredToken)).toBe(true);
    });

    it('should identify non-expired tokens', () => {
        const validToken = createToken({ exp: Math.floor(Date.now() / 1000) + 3600 });
        expect(isJwtExpired(validToken)).toBe(false);
    });

    it('should identify tokens close to expiration with buffer', () => {
        const soonExpiredToken = createToken({ exp: Math.floor(Date.now() / 1000) + 5 });
        expect(isJwtExpired(soonExpiredToken, 10)).toBe(true);
    });

    it('should treat tokens without exp as expired (safe default)', () => {
        const tokenWithoutExp = createToken({ id: '123' });
        expect(isJwtExpired(tokenWithoutExp)).toBe(true);
    });
});
