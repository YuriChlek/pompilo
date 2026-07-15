import { describe, it, expect } from 'vitest';
import { buildRefreshHeaders, hasCookie, appendSetCookies } from '@/features/module-auth/server/session-http';
import { COOKIE_NAMES } from '@/features/module-auth/enums/auth.enums';
import { NextRequest, NextResponse } from 'next/server';

describe('session-http utilities', () => {
    describe('buildRefreshHeaders', () => {
        it('should forward deviceId in cookie headers when present', () => {
            const request = {
                cookies: {
                    get: (name: string) => {
                        if (name === COOKIE_NAMES.DEVICE_ID) {
                            return { name: COOKIE_NAMES.DEVICE_ID, value: 'device-123' };
                        }
                        if (name === COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN) {
                            return { name: COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN, value: 'refresh-456' };
                        }
                        return null;
                    },
                },
                headers: {
                    get: (name: string) => {
                        if (name === 'user-agent') return 'test-ua';
                        if (name === 'x-forwarded-for') return '1.2.3.4';
                        return null;
                    },
                },
            } as unknown as NextRequest;

            const headers = buildRefreshHeaders(request, 'customer') as Record<string, string>;

            expect(headers.cookie).toContain(`${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=refresh-456`);
            expect(headers.cookie).toContain(`${COOKIE_NAMES.DEVICE_ID}=device-123`);
            expect(headers['user-agent']).toBe('test-ua');
            expect(headers['x-forwarded-for']).toBe('1.2.3.4');
        });

        it('should NOT forward deviceId in cookie headers when absent', () => {
            const request = {
                cookies: {
                    get: (name: string) => {
                        if (name === COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN) {
                            return { name: COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN, value: 'refresh-456' };
                        }
                        return null;
                    },
                },
                headers: {
                    get: () => null,
                },
            } as unknown as NextRequest;

            const headers = buildRefreshHeaders(request, 'customer') as Record<string, string>;

            expect(headers.cookie).toBe(`${COOKIE_NAMES.CUSTOMER_REFRESH_TOKEN}=refresh-456`);
        });
    });

    describe('hasCookie', () => {
        it('should return true when cookie is present', () => {
            const request = {
                cookies: {
                    get: (name: string) => name === COOKIE_NAMES.DEVICE_ID ? { value: 'some-value' } : null,
                },
            } as unknown as NextRequest;

            expect(hasCookie(request, COOKIE_NAMES.DEVICE_ID)).toBe(true);
        });

        it('should return false when cookie is missing', () => {
            const request = {
                cookies: {
                    get: () => null,
                },
            } as unknown as NextRequest;

            expect(hasCookie(request, COOKIE_NAMES.DEVICE_ID)).toBe(false);
        });
    });

    describe('appendSetCookies', () => {
        it('should append set-cookie headers to response', () => {
            const response = NextResponse.next();
            appendSetCookies(response, ['cookie1=val1', 'cookie2=val2']);
            
            const setCookies = response.headers.getSetCookie();
            expect(setCookies).toContain('cookie1=val1');
            expect(setCookies).toContain('cookie2=val2');
        });
    });
});
