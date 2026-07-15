import { describe, it, expect } from 'vitest';
import { mergeSetCookiesIntoCookieHeader } from '@/features/module-auth/lib/cookie-merge';

describe('mergeSetCookiesIntoCookieHeader', () => {
    it('should merge multiple Set-Cookie headers into existing cookie header', () => {
        const currentCookie = 'a=1; b=2';
        const setCookies = ['c=3; Path=/', 'a=4; HttpOnly'];
        
        const result = mergeSetCookiesIntoCookieHeader(currentCookie, setCookies);
        
        expect(result).toBe('a=4; b=2; c=3');
    });

    it('should handle empty current cookie header', () => {
        const currentCookie = '';
        const setCookies = ['a=1; Path=/', 'b=2'];
        
        const result = mergeSetCookiesIntoCookieHeader(currentCookie, setCookies);
        
        expect(result).toBe('a=1; b=2');
    });

    it('should handle null/undefined current cookie header', () => {
        const setCookies = ['a=1'];
        
        expect(mergeSetCookiesIntoCookieHeader(null, setCookies)).toBe('a=1');
        expect(mergeSetCookiesIntoCookieHeader(undefined, setCookies)).toBe('a=1');
    });

    it('should ignore attributes in Set-Cookie headers', () => {
        const setCookies = ['token=val; Domain=example.com; Secure; HttpOnly; Max-Age=3600'];
        
        const result = mergeSetCookiesIntoCookieHeader('', setCookies);
        
        expect(result).toBe('token=val');
    });

    it('should overwrite existing cookies with new values from Set-Cookie', () => {
        const currentCookie = 'session=old; theme=dark';
        const setCookies = ['session=new; Path=/'];
        
        const result = mergeSetCookiesIntoCookieHeader(currentCookie, setCookies);
        
        expect(result).toBe('session=new; theme=dark');
    });

    it('should handle cookies with equals signs in values', () => {
        const currentCookie = 'data=base64==';
        const setCookies = ['other=foo=bar'];
        
        const result = mergeSetCookiesIntoCookieHeader(currentCookie, setCookies);
        
        expect(result).toBe('data=base64==; other=foo=bar');
    });

    it('should handle malformed or empty Set-Cookie strings', () => {
        const setCookies = ['', ';', 'nameOnly', '=valueOnly'];
        
        const result = mergeSetCookiesIntoCookieHeader('a=1', setCookies);
        
        // '' -> ignored
        // ';' -> mainPart is '', ignored
        // 'nameOnly' -> mainPart is 'nameOnly', [name, ...valueParts] = ['nameOnly'], valueParts.join('=') is '', result 'nameOnly='
        // '=valueOnly' -> mainPart is '=valueOnly', [name, ...valueParts] = ['', 'valueOnly'], rawName is '', ignored
        expect(result).toBe('a=1; nameOnly=');
    });
});
